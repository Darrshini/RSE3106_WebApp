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
 * live in the client so the server stays stateless. Everything about the dotted
 * lines is driven by the model's SEGMENTATION MASKS (output1), never their
 * bounding boxes: we threshold each instance mask, run PCA on its pixels for the
 * line fit, take the vanishing point of the dashed boundaries as the heading
 * toward the far kerb, and hand the client the mask bitmap itself to draw. We
 * never steer toward the pedestrian light — the pedestrian crosses to the
 * opposite kerb, not the light; the light is only a weak agreement check.
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

// `rotate` is degrees CLOCKWISE, applied before anything else. The Pi's Camera
// Module is mounted sideways, so its frames arrive 90 deg CCW of upright and the
// server un-rotates them here, at the single point where it decodes the JPEG --
// so this costs nothing extra. Everything downstream (letterbox, w/h, origRaw,
// and therefore every box/mask/angle we return) is then in UPRIGHT coordinates,
// which is the same space the browser draws in after rotating the frame on its
// canvas. A webcam or an uploaded video is already upright, so those callers
// pass rotate=0 and this whole path is a no-op for them.
async function preprocess(buf, rotate) {
    const src = () => (rotate ? sharp(buf).rotate(rotate) : sharp(buf));

    // Take w/h from the ROTATED raw buffer, not from metadata() -- metadata()
    // reports the JPEG's on-disk dimensions, which are still the un-rotated ones.
    const { data: origRaw, info } =
        await src().removeAlpha().raw().toBuffer({ resolveWithObject: true });  // also the HSV colour read
    const w = info.width, h = info.height;

    const { scale, nw, nh, padX, padY } = letterbox(w, h);
    const lb = await src().removeAlpha()
        .resize(nw, nh, { fit: 'fill' })
        .extend({ top: padY, bottom: INPUT - nh - padY, left: padX, right: INPUT - nw - padX,
                  background: { r: 114, g: 114, b: 114 } })
        .raw().toBuffer();
    const area = INPUT * INPUT, data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        data[i] = lb[i*3] / 255; data[i+area] = lb[i*3+1] / 255; data[i+2*area] = lb[i*3+2] / 255;
    }
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
    // Report green-man and red presence INDEPENDENTLY -- the client needs both:
    //   green only     = constant WALK (no countdown numeral)
    //   green AND red   = clearance (flashing green man + red countdown) ON-phase
    //   red only        = clearance OFF-blink OR constant DON'T-WALK (client times it out at 2s)
    const gMin = Math.max(15, 0.012*boxPx), rMin = Math.max(15, 0.020*boxPx);
    const g = green >= gMin, r = red >= rMin;
    return { green: g, red: r,
             state: g ? (r ? 'GREENRED' : 'GREEN') : (r ? 'RED' : 'NONE'),
             greenCount: green, redCount: red, lit };
}

// ONE dotted-line instance, taken entirely from its SEGMENTATION MASK (not its
// box). mask(px,py) = sigmoid(coeffs . protos); protos (output1) are at 160x160
// in the letterboxed frame, so sigmoid(s)>0.5 <=> s>0 (no exp needed). We keep
// the box only as the window to scan the prototype grid (Ultralytics crops masks
// to the box anyway). From the thresholded mask pixels we derive: the PCA line
// axis + elongation (a high ratio = a genuine line, not a blob), the pixel
// centroid, the lowest mask pixel (mask-based end-of-crossing), and a compact
// bit-packed bitmap of the mask (+ its box in ORIGINAL coords) for the client to
// draw the actual segmentation.
function dottedMask(P, det, scale, padX, padY) {
    const MW = 160, MA = MW * MW, c = det.coeffs;
    const lx1 = det.x1*scale+padX, ly1 = det.y1*scale+padY, lx2 = det.x2*scale+padX, ly2 = det.y2*scale+padY;
    const mx1 = Math.max(0,Math.floor(lx1/4)), my1 = Math.max(0,Math.floor(ly1/4));
    const mx2 = Math.min(MW,Math.ceil(lx2/4)), my2 = Math.min(MW,Math.ceil(ly2/4));
    const mw = mx2-mx1, mh = my2-my1;
    if (mw < 1 || mh < 1) return null;
    const bits = new Uint8Array((mw*mh + 7) >> 3);            // 1 bit per prototype cell, row-major
    const pts = []; let maxY = -1;
    for (let my=my1; my<my2; my++) for (let mx=mx1; mx<mx2; mx++){
        let s=0; const base=my*MW+mx; for (let k=0;k<32;k++) s += c[k]*P[k*MA+base];
        if (s>0) {                                            // pixel is inside the segmentation mask
            const idx = (my-my1)*mw + (mx-mx1); bits[idx>>3] |= (1 << (idx&7));
            const ox=((mx+0.5)*4-padX)/scale, oy=((my+0.5)*4-padY)/scale;   // cell centre -> original
            pts.push({ x:ox, y:oy }); if (oy>maxY) maxY=oy;
        }
    }
    if (pts.length < 6) return null;
    let cx=0,cy=0; for(const p of pts){cx+=p.x;cy+=p.y;} cx/=pts.length; cy/=pts.length;
    let sxx=0,sxy=0,syy=0; for(const p of pts){const dx=p.x-cx,dy=p.y-cy; sxx+=dx*dx;sxy+=dx*dy;syy+=dy*dy;}
    sxx/=pts.length; sxy/=pts.length; syy/=pts.length;
    const tr=sxx+syy, D=sxx*syy-sxy*sxy, disc=Math.sqrt(Math.max(0,tr*tr/4-D));
    const l1=tr/2+disc, l2=tr/2-disc;
    const th=0.5*Math.atan2(2*sxy, sxx-syy); let dx=Math.cos(th), dy=Math.sin(th);
    if (dy>0){ dx=-dx; dy=-dy; }                              // orient axis toward the far side (up)
    // client render payload: the mask bitmap + its box in ORIGINAL coords (cell EDGES)
    const box = [ (mx1*4-padX)/scale, (my1*4-padY)/scale, (mx2*4-padX)/scale, (my2*4-padY)/scale ];
    return { cx, cy, dx, dy, count:pts.length, elong:(l2>1e-6? l1/l2 : 999), maxY,
             client: { box, mw, mh, data: Buffer.from(bits).toString('base64') } };
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

function corridor(lines, light, w, h) {
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
            const t = ((B.cx-A.cx)*B.dy - (B.cy-A.cy)*B.dx) / den;
            const ix = A.cx + t*A.dx, iy = A.cy + t*A.dy;
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

async function infer(buf, opts) {
    const rotate = (opts && opts.rotate) || 0;
    const s = await load();
    const pre = await preprocess(buf, rotate);
    const out = await s.run({ [s.inputNames[0]]: pre.tensor });
    const proto = out[s.outputNames[1]].data;                 // output1: 32 mask prototypes @160x160
    const dets = decode(out[s.outputNames[0]], pre.scale, pre.padX, pre.padY, pre.w, pre.h);
    const dottedDets = dets.filter(d=>d.cls===DOTTED);
    const lightDets = dets.filter(d=>d.cls===LIGHT).sort((a,b)=>b.score-a.score);
    const light = lightDets[0] || null;

    // Segmentation mask per dotted line -- this (not the box) drives everything.
    const masks = [];
    for (const det of dottedDets) {
        const m = dottedMask(proto, det, pre.scale, pre.padX, pre.padY);
        if (m) { m.conf = det.score; masks.push(m); }
    }
    const lines = masks.filter(m => m.count >= 8 && m.elong >= 2.0)   // genuine line segments
                       .map(m => ({ cx:m.cx, cy:m.cy, dx:m.dx, dy:m.dy, count:m.count }));

    // Read the state of EVERY pedestrian light in frame, not just the strongest.
    // The browser's SCANNING logic (which used to run on pedestrian.onnx's
    // 'traffic-light' boxes) needs all of them to offer a left/right post choice
    // when two are visible. `light` stays the primary (highest-conf) one, which is
    // what the WAITING/CROSSING light-state decisions read.
    const lightsOut = lightDets.map(l => {
        const ls = lightState(pre.origRaw, pre.w, pre.h, l);
        return { box:[l.x1,l.y1,l.x2,l.y2], conf:l.score,
                 green: ls.green, red: ls.red, state: ls.state,
                 areaFrac: ((l.x2-l.x1)*(l.y2-l.y1))/(pre.w*pre.h) };
    });
    const lightOut = lightsOut[0] || null;
    const cor = corridor(lines, light, pre.w, pre.h);
    const lowestMaskY = masks.length ? Math.max(...masks.map(m=>m.maxY)) : null;   // mask-based, not box
    return {
        w: pre.w, h: pre.h,
        light: lightOut,
        lights: lightsOut,
        dotted: masks.map(m => ({ mask:m.client, conf:m.conf })),   // segmentation bitmap, not a box
        corridor: cor,
        signals: {
            corridorAhead: masks.length > 0,
            lightAreaFrac: lightOut ? lightOut.areaFrac : 0,
            nDashes: masks.length,
            lowestDashYFrac: lowestMaskY != null ? lowestMaskY / pre.h : null
        }
    };
}

module.exports = { load, infer };
