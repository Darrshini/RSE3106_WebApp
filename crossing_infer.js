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
 * live in the client so the server stays stateless. Masks (output1) aren't
 * needed for the direction — the dotted-line box centroids give the corridor
 * axis — so v1 uses boxes and skips the per-instance mask matmul for speed.
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
    const d=out0.data, na=out0.dims[2], nc=CLASSES.length, boxes=[];
    for (let a=0;a<na;a++){
        let best=0,bc=0; for(let c=0;c<nc;c++){ const s=d[(4+c)*na+a]; if(s>best){best=s;bc=c;} }
        if (best<CONF) continue;
        const cx=d[a], cy=d[na+a], bw=d[2*na+a], bh=d[3*na+a];
        const clamp=(v,m)=>Math.max(0,Math.min(m,v));
        boxes.push({ x1:clamp((cx-bw/2-padX)/scale,w), y1:clamp((cy-bh/2-padY)/scale,h),
                     x2:clamp((cx+bw/2-padX)/scale,w), y2:clamp((cy+bh/2-padY)/scale,h), score:best, cls:bc });
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

// Heading from the user (bottom-centre) toward the goal (the far end of the
// crossing). Base heading = user -> target, where target is the pedestrian
// light if it's ahead, else the farthest (highest) dash. When enough dashes
// recede into the distance, refine the heading with their principal axis so
// the arrow runs *parallel to the dashed lines* — but only if that axis agrees
// with the user->target direction (guards against a left/right dash pair
// producing an across-the-road axis).
function corridor(dotted, light, w, h) {
    const user = { x: w / 2, y: h * 0.98 };
    const pts = dotted.map(d => ({ x:(d.x1+d.x2)/2, y:(d.y1+d.y2)/2 }))
                      .filter(p => p.x > w*0.04 && p.x < w*0.96);
    if (!pts.length && !light) return { has: false };

    let target = null;
    if (light) { const lc={ x:(light.x1+light.x2)/2, y:(light.y1+light.y2)/2 }; if (lc.y < h*0.85) target = lc; }
    if (!target && pts.length) target = pts.reduce((a,b)=>a.y<b.y?a:b);   // farthest dash
    if (!target) return { has: false };

    let dx = target.x-user.x, dy = target.y-user.y; const L = Math.hypot(dx,dy)||1; dx/=L; dy/=L;

    if (pts.length >= 3) {
        const ys = pts.map(p=>p.y), yspread = (Math.max(...ys)-Math.min(...ys))/h;
        if (yspread > 0.12) {                                // dashes recede -> their axis is meaningful
            const n=pts.length; let mx=0,my=0; for(const p of pts){mx+=p.x;my+=p.y;} mx/=n; my/=n;
            let sxx=0,sxy=0,syy=0; for(const p of pts){const ex=p.x-mx,ey=p.y-my; sxx+=ex*ex;sxy+=ex*ey;syy+=ey*ey;}
            const th=0.5*Math.atan2(2*sxy, sxx-syy); let ax=Math.cos(th), ay=Math.sin(th);
            if (ay>0){ ax=-ax; ay=-ay; }
            if (ax*dx + ay*dy > 0.5) { dx=ax; dy=ay; }       // adopt axis only if it agrees with user->target
        }
    }
    const dist = Math.hypot(target.x-user.x, target.y-user.y);
    return { has: true, near: user, far: { x:user.x+dx*dist, y:user.y+dy*dist },
             target, angleDeg: Math.atan2(dy,dx)*180/Math.PI };
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
    const cor = corridor(dotted, light, pre.w, pre.h);
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
