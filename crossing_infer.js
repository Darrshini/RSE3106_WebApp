/**
 * crossing_infer.js — server-side perception for NavAssist crossing navigation.
 *
 * Runs the YOLO11-seg model (classes: 'dotted line', 'pedestrian light') via
 * onnxruntime-node, and turns raw detections into the high-level signals the
 * client FSM needs, all statelessly per frame:
 *   - pedestrian-light box + GREEN/RED/UNKNOWN state (HSV on the lit pixels)
 *   - the crossing-corridor direction vector (near -> far) from the dotted lines
 *   - per-frame end-of-crossing signals (dashes-ahead + light proximity)
 *
 * The temporal decisions (fresh red->green cross timing, "reached the end")
 * live in the client so the server stays stateless. The crossing DIRECTION comes
 * purely from the segmented dotted lines (output1 masks): each dotted-line run is
 * elongated ALONG the crossing, so its instance-mask principal axis IS the
 * heading toward the far kerb. We never steer toward the pedestrian light — the
 * pedestrian crosses to the opposite kerb, not to the light; the light only
 * serves as a weak agreement check.
 */
const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');

const MODEL_PATH = path.join(__dirname, 'public', 'models', 'crossing_seg.onnx');
const INPUT = 640;
const CONF = 0.30;
const IOU = 0.50;
const CLASSES = ['dotted line', 'pedestrian light'];
const DOTTED = 0, LIGHT = 1;

let session = null;
async function load() {
    if (!session) {
        session = await ort.InferenceSession.create(MODEL_PATH);
        console.log('[infer] model loaded:', MODEL_PATH);
    }
    return session;
}

function letterbox(w, h) {
    const scale = Math.min(INPUT / w, INPUT / h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    return { scale, nw, nh, padX: Math.floor((INPUT - nw) / 2), padY: Math.floor((INPUT - nh) / 2) };
}

async function preprocess(buf) {
    const meta = await sharp(buf).metadata();
    const w = meta.width, h = meta.height;
    const { scale, nw, nh, padX, padY } = letterbox(w, h);
    const lb = await sharp(buf).removeAlpha()
        .resize(nw, nh, { fit: 'fill' })
        .extend({ top: padY, bottom: INPUT - nh - padY, left: padX, right: INPUT - nw - padX,
                  background: { r: 114, g: 114, b: 114 } })
        .raw().toBuffer();
    const area = INPUT * INPUT, data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        data[i] = lb[i*3] / 255; data[i+area] = lb[i*3+1] / 255; data[i+2*area] = lb[i*3+2] / 255;
    }
    const origRaw = await sharp(buf).removeAlpha().raw().toBuffer();   // for the HSV colour read
    return { tensor: new ort.Tensor('float32', data, [1,3,INPUT,INPUT]), scale, padX, padY, w, h, origRaw };
}

function iou(a, b) {
    const x1=Math.max(a.x1,b.x1), y1=Math.max(a.y1,b.y1), x2=Math.min(a.x2,b.x2), y2=Math.min(a.y2,b.y2);
    const inter=Math.max(0,x2-x1)*Math.max(0,y2-y1);
    return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter + 1e-6);
}
function nms(boxes, thr) {
    boxes.sort((a,b)=>b.score-a.score); const keep=[], dead=new Array(boxes.length).fill(false);
    for (let i=0;i<boxes.length;i++){ if(dead[i])continue; keep.push(boxes[i]);
        for(let j=i+1;j<boxes.length;j++) if(!dead[j] && boxes[i].cls===boxes[j].cls && iou(boxes[i],boxes[j])>thr) dead[j]=true; }
    return keep;
}
function decode(out0, scale, padX, padY, w, h) {
    const d=out0.data, na=out0.dims[2], nc=CLASSES.length, nm=out0.dims[1]-4-nc, boxes=[];
    for (let a=0;a<na;a++){
        let best=0,bc=0; for(let c=0;c<nc;c++){ const s=d[(4+c)*na+a]; if(s>best){best=s;bc=c;} }
        if (best<CONF) continue;
        const cx=d[a], cy=d[na+a], bw=d[2*na+a], bh=d[3*na+a];
        const clamp=(v,m)=>Math.max(0,Math.min(m,v));
        const coeffs=new Float32Array(nm); for(let k=0;k<nm;k++) coeffs[k]=d[(4+nc+k)*na+a];  // 32 mask coeffs
        boxes.push({ x1:clamp((cx-bw/2-padX)/scale,w), y1:clamp((cy-bh/2-padY)/scale,h),
                     x2:clamp((cx+bw/2-padX)/scale,w), y2:clamp((cy+bh/2-padY)/scale,h), score:best, cls:bc, coeffs });
    }
    return nms(boxes, IOU);
}

function rgb2hsv(r,g,b){ r/=255;g/=255;b/=255; const mx=Math.max(r,g,b),mn=Math.min(r,g,b),df=mx-mn;
    let h=0; if(df){ if(mx===r)h=60*(((g-b)/df)%6); else if(mx===g)h=60*((b-r)/df+2); else h=60*((r-g)/df+4); }
    if(h<0)h+=360; return [h, mx?df/mx:0, mx]; }

// GREEN/RED/UNKNOWN from the *lit* (bright + saturated) pixels inside the light box.
// Key nuance seen in real footage: during the WALK phase the box shows BOTH the
// lit green man AND a red/orange remaining-seconds countdown -> "green vs red"
// counting fails. The green man only ever lights during the walk phase, so its
// *presence* is the decision. Also the green man renders green->teal/cyan on this
// camera (hue ~90-210), not pure green.
function lightState(raw, w, h, box) {
    const x1=Math.max(0,Math.floor(box.x1)), y1=Math.max(0,Math.floor(box.y1));
    const x2=Math.min(w,Math.ceil(box.x2)), y2=Math.min(h,Math.ceil(box.y2));
    const boxPx = Math.max(1, (x2-x1)*(y2-y1));
    let green=0, red=0, lit=0;
    for (let y=y1;y<y2;y++) for (let x=x1;x<x2;x++){
        const i=(y*w+x)*3, [hh,s,v]=rgb2hsv(raw[i],raw[i+1],raw[i+2]);
        if (v>0.45 && s>0.35){ lit++;
            if (hh>=90 && hh<=210) green++;        // green man: green -> teal/cyan
            else if (hh<=22 || hh>=335) red++;     // red man / red pixels
        }
    }
    const gMin = Math.max(15, 0.012*boxPx), rMin = Math.max(15, 0.020*boxPx);
    let state='UNKNOWN';
    if (green >= gMin) state='GREEN';              // green man lit = walk (even with a red countdown)
    else if (red >= rMin) state='RED';
    return { state, green, red, lit };
}

// Principal axis of ONE dotted-line instance from its segmentation mask.
// mask(px,py) = sigmoid(coeffs . protos); protos (output1) are at 160x160 in the
// letterboxed frame, so sigmoid(s)>0.5 <=> s>0 (no exp needed for the test).
// Returns the pixels (in ORIGINAL image coords) plus the covariance eigen-axis
// and an elongation ratio; a high ratio means it's a genuine line (not a blob).
function maskAxis(P, det, scale, padX, padY) {
    const MW = 160, MA = MW * MW, c = det.coeffs;
    const lx1 = det.x1*scale+padX, ly1 = det.y1*scale+padY, lx2 = det.x2*scale+padX, ly2 = det.y2*scale+padY;
    const mx1 = Math.max(0,Math.floor(lx1/4)), my1 = Math.max(0,Math.floor(ly1/4));
    const mx2 = Math.min(MW,Math.ceil(lx2/4)), my2 = Math.min(MW,Math.ceil(ly2/4));
    const pts = [];
    for (let my=my1; my<my2; my++) for (let mx=mx1; mx<mx2; mx++){
        let s=0; const base=my*MW+mx; for (let k=0;k<32;k++) s += c[k]*P[k*MA+base];
        if (s>0) pts.push({ x:((mx+0.5)*4-padX)/scale, y:((my+0.5)*4-padY)/scale });   // sigmoid>0.5
    }
    if (pts.length < 6) return null;
    let cx=0,cy=0; for(const p of pts){cx+=p.x;cy+=p.y;} cx/=pts.length; cy/=pts.length;
    let sxx=0,sxy=0,syy=0; for(const p of pts){const dx=p.x-cx,dy=p.y-cy; sxx+=dx*dx;sxy+=dx*dy;syy+=dy*dy;}
    sxx/=pts.length; sxy/=pts.length; syy/=pts.length;
    const tr=sxx+syy, D=sxx*syy-sxy*sxy, disc=Math.sqrt(Math.max(0,tr*tr/4-D));
    const l1=tr/2+disc, l2=tr/2-disc;
    const th=0.5*Math.atan2(2*sxy, sxx-syy); let dx=Math.cos(th), dy=Math.sin(th);
    if (dy>0){ dx=-dx; dy=-dy; }                              // orient each axis toward the far side (up)
    return { dx, dy, count:pts.length, elong:(l2>1e-6? l1/l2 : 999), pts };
}

// Crossing DIRECTION from the segmented dotted lines only (never the light).
// The two dashed boundary lines are parallel on the ground but, in perspective,
// they CONVERGE to a vanishing point — the far end the crossing leads to. So we
// fit each dotted-line instance to a line, take the vanishing point as the
// robust average of their pairwise intersections, and the heading is
// user(bottom-centre) -> vanishing point. That is the corridor centreline: it
// runs *between* the boundary lines toward the far kerb, which is why averaging
// the individually perspective-tilted line angles is wrong. Falls back to a
// single line's own direction when only one boundary is visible. The light is
// only a weak sanity flag.
function median(arr){ const s=[...arr].sort((a,b)=>a-b); return s[(s.length-1)>>1]; }

function corridor(dotted, P, scale, padX, padY, light, w, h) {
    const lines = [];
    for (const det of dotted) {
        const m = maskAxis(P, det, scale, padX, padY);
        if (!m || m.count < 8 || m.elong < 2.0) continue;    // must be a genuine line segment
        let cx=0, cy=0; for (const p of m.pts){ cx+=p.x; cy+=p.y; } cx/=m.pts.length; cy/=m.pts.length;
        lines.push({ px:cx, py:cy, dx:m.dx, dy:m.dy, count:m.count });
    }
    if (!lines.length) return { has:false };

    const user = { x: w/2, y: h*0.98 };
    let hx, hy;
    if (lines.length === 1) {
        hx = lines[0].dx; hy = lines[0].dy;                  // one boundary -> walk parallel to it
    } else {
        const xs=[], ys=[];                                  // vanishing point = pairwise intersections
        for (let i=0;i<lines.length;i++) for (let j=i+1;j<lines.length;j++){
            const A=lines[i], B=lines[j], den=A.dx*B.dy - A.dy*B.dx;
            if (Math.abs(den) < 1e-3) continue;              // ~parallel -> no stable intersection
            const t = ((B.px-A.px)*B.dy - (B.py-A.py)*B.dx) / den;
            const ix = A.px + t*A.dx, iy = A.py + t*A.dy;
            if (iy > user.y || iy < -3*h || ix < -4*w || ix > 5*w) continue;   // implausible VP
            xs.push(ix); ys.push(iy);
        }
        if (xs.length) { hx = median(xs)-user.x; hy = median(ys)-user.y; }     // head toward the VP
        else { hx=0; hy=0; for (const l of lines){ hx+=l.dx; hy+=l.dy; } }     // parallel -> mean direction
    }
    const L=Math.hypot(hx,hy)||1; hx/=L; hy/=L; if (hy>0){ hx=-hx; hy=-hy; }   // orient toward the far side (up)
    if (hy > -0.15) return { has:false };                    // basically across the road -> unusable

    const far = { x: user.x + hx*h*0.55, y: user.y + hy*h*0.55 };   // arrow up the corridor from the user
    let lightAgrees = null;
    if (light){ const lc={x:(light.x1+light.x2)/2, y:(light.y1+light.y2)/2};
        const ux=lc.x-user.x, uy=lc.y-user.y, ll=Math.hypot(ux,uy)||1;
        lightAgrees = (hx*ux+hy*uy)/ll > 0.3; }
    return { has:true, near:user, far, angleDeg: Math.atan2(hy,hx)*180/Math.PI, lightAgrees };
}

async function infer(buf) {
    const s = await load();
    const pre = await preprocess(buf);
    const out = await s.run({ [s.inputNames[0]]: pre.tensor });
    const dets = decode(out[s.outputNames[0]], pre.scale, pre.padX, pre.padY, pre.w, pre.h);
    const dotted = dets.filter(d=>d.cls===DOTTED);
    const lights = dets.filter(d=>d.cls===LIGHT).sort((a,b)=>b.score-a.score);
    const light = lights[0] || null;

    let lightOut = null;
    if (light) {
        const ls = lightState(pre.origRaw, pre.w, pre.h, light);
        lightOut = { box:[light.x1,light.y1,light.x2,light.y2], conf:light.score, state:ls.state,
                     areaFrac: ((light.x2-light.x1)*(light.y2-light.y1))/(pre.w*pre.h) };
    }
    const cor = corridor(dotted, out[s.outputNames[1]].data, pre.scale, pre.padX, pre.padY, light, pre.w, pre.h);
    const lowestDashY = dotted.length ? Math.max(...dotted.map(d=>d.y2)) : null;
    return {
        w: pre.w, h: pre.h,
        light: lightOut,
        dotted: dotted.map(d=>({ box:[d.x1,d.y1,d.x2,d.y2], conf:d.score })),
        corridor: cor,
        signals: {
            corridorAhead: dotted.length > 0,
            lightAreaFrac: lightOut ? lightOut.areaFrac : 0,
            nDashes: dotted.length,
            lowestDashYFrac: lowestDashY != null ? lowestDashY / pre.h : null
        }
    };
}

module.exports = { load, infer };
