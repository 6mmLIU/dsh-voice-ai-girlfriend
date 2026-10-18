// ASMR stage: plays the server's binaural audio and animates hands, felt and
// lights in sync with its cue timeline.
//
//   - lights follow the REAL left/right loudness of what is playing (Web Audio
//     analysers on each channel), so a whisper in the left ear lights the left
//   - cues from the server drive the hands: alternating taps, rubbing the felt
//     with alternating hands, a brush sweeping across, alternating light pulses
//   - real footage: asmr/media/idle/ loops underneath, and every cue that
//     carries a clip url plays that exact take at its cue time (preloaded as
//     soon as the item arrives, so it starts on time)
//   - the drawn hands/felt are only a placeholder for when no footage exists
//
// URL flags: ?debug=1 shows a HUD, ?caption=0 hides the reply caption,
// ?lights=0 turns off the light overlay.
'use strict'

const W = 1080
const H = 1920
const params = new URLSearchParams(location.search)
const DEBUG = params.get('debug') === '1'
const SHOW_CAPTION = params.get('caption') !== '0'
const SHOW_LIGHTS = params.get('lights') !== '0'

const scene = document.getElementById('scene').getContext('2d')
const lightsCtx = document.getElementById('lights').getContext('2d')
const baseClip = document.getElementById('baseClip')
const stageEl = document.getElementById('stage')
const lightsEl = document.getElementById('lights')
const caption = document.getElementById('caption')
const hud = document.getElementById('hud')
if (DEBUG) hud.classList.add('on')

const now = () => performance.now() / 1000
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))
const ease = (t) => 0.5 - 0.5 * Math.cos(Math.PI * clamp(t, 0, 1))

// ── Audio ────────────────────────────────────────────────────────────────
let ac = null
let analyserL = null
let analyserR = null
const bufL = new Float32Array(1024)
const bufR = new Float32Array(1024)
const level = { L: 0, R: 0 }

function initAudio () {
  ac = new AudioContext({ sampleRate: 48000 })
  const splitter = ac.createChannelSplitter(2)
  analyserL = ac.createAnalyser()
  analyserR = ac.createAnalyser()
  analyserL.fftSize = analyserR.fftSize = 1024
  splitter.connect(analyserL, 0)
  splitter.connect(analyserR, 1)
  ac.__splitter = splitter
}

function rms (analyser, buf) {
  analyser.getFloatTimeDomainData(buf)
  let sum = 0
  for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i]
  return Math.sqrt(sum / buf.length)
}

function readLevels () {
  if (!analyserL) return
  // Whisper RMS sits around 0.02-0.08; map to 0..1 with a soft curve.
  const map = (v) => clamp(Math.pow(v * 9, 0.7), 0, 1)
  const l = map(rms(analyserL, bufL))
  const r = map(rms(analyserR, bufR))
  level.L += (l - level.L) * (l > level.L ? 0.35 : 0.06)
  level.R += (r - level.R) * (r > level.R ? 0.35 : 0.06)
}

// ── Playback queue + cue timeline ─────────────────────────────────────────
const queue = []
let playing = null // { item, start (stage clock), fired:Set }
const actions = [] // { name, side, start, dur, az0, az1 }

async function playNext () {
  if (playing || queue.length === 0 || !ac) return
  const item = queue.shift()
  try {
    const resp = await fetch(item.url)
    const audio = await ac.decodeAudioData(await resp.arrayBuffer())
    const src = ac.createBufferSource()
    src.buffer = audio
    src.connect(ac.destination)
    src.connect(ac.__splitter)
    // Start when the idle loop will be exactly on its first frame at the
    // item's first clip, so the cut from idle into the action is invisible.
    const delay = planIdleStart(item)
    playing = { item, start: now() + delay, fired: new Set() }
    src.start(ac.currentTime + delay)
    src.onended = () => {
      playing = null
      hideCaption()
      playNext()
    }
    showCaption(item)
  } catch (err) {
    console.error('[asmr] play failed', err)
    playing = null
    playNext()
  }
}

function fireCues () {
  if (!playing) return
  const t = now() - playing.start
  playing.item.cues.forEach((cue, i) => {
    if (playing.fired.has(i) || cue.t > t) return
    playing.fired.add(i)
    if (cue.kind === 'clip') {
      playClipCue(cue, `${playing.item.id}:${i}`)
    } else if (cue.kind === 'speech') {
      actions.push({ name: 'whisper', side: cue.pos, start: playing.start + cue.t, dur: cue.dur })
    } else {
      actions.push({ name: cue.name, side: cue.side, start: playing.start + cue.t, dur: cue.dur, az0: cue.az0, az1: cue.az1 })
    }
  })
}

function activeActions (t) {
  for (let i = actions.length - 1; i >= 0; i--) {
    if (t > actions[i].start + actions[i].dur + 1.5) actions.splice(i, 1)
  }
  return actions.filter((a) => t >= a.start && t <= a.start + a.dur)
}

// ── Caption ──────────────────────────────────────────────────────────────
let captionTimer = 0
function showCaption (item) {
  clearTimeout(captionTimer)
  if (!SHOW_CAPTION || !item.display) {
    caption.classList.remove('on')
    return
  }
  caption.querySelector('.who').textContent = item.user ? `回复 @${item.user}` : ''
  caption.querySelector('.what').textContent = item.display
  caption.classList.add('on')
}
function hideCaption () {
  clearTimeout(captionTimer)
  captionTimer = setTimeout(() => caption.classList.remove('on'), 800)
}

// ── Footage: today's outfit (idle video or still) + action clips ──────────
let clips = {}
let idleKey = ''
const baseImg = document.getElementById('baseImg')

function setIdle (videos, stills) {
  const pick = (list) => list[Math.floor(Math.random() * list.length)]
  // A prebuilt *_loop take (forward + backward takes joined) loops on one
  // element with no source switch, so the idle never blinks.
  const loops = videos.filter((u) => u.includes('_loop.'))
  if (loops.length) videos = loops
  if (videos.length) {
    baseImg.classList.remove('on')
    baseClip.src = pick(videos)
    baseClip.onended = () => { baseClip.src = pick(videos); baseClip.play().catch(() => {}) }
    baseClip.loop = videos.length === 1
    baseClip.play().catch(() => {})
    baseClip.classList.add('on')
  } else if (stills.length) {
    // No idle video yet: a still that slowly "breathes" stands in for it.
    baseClip.classList.remove('on')
    baseClip.pause()
    baseImg.src = pick(stills)
    baseImg.classList.add('on')
  } else {
    baseClip.classList.remove('on')
    baseImg.classList.remove('on')
  }
}

async function loadClips () {
  let data = {}
  try {
    data = await (await fetch('/api/media')).json()
  } catch { return }
  clips = data.clips || {}
  const videos = clips.idle || []
  const stills = data.stills || []
  // Re-pick the idle only when the outfit or its files change.
  const key = JSON.stringify([data.outfit, videos, stills])
  if (key !== idleKey) {
    idleKey = key
    setIdle(videos, stills)
  }
}
const clipPool = new Map() // url -> preloaded <video>
let activeClips = [] // { video, until }

function makeVideo (url) {
  const v = document.createElement('video')
  v.className = 'clip'
  v.muted = true
  v.playsInline = true
  v.preload = 'auto'
  v.src = url
  stageEl.insertBefore(v, lightsEl)
  return v
}

function preloadClips (item) {
  // One preloaded <video> per clip cue (a long action chains several takes,
  // often of the same file), keyed by item and cue index.
  item.cues.forEach((cue, i) => {
    if (cue.kind === 'clip') clipPool.set(`${item.id}:${i}`, makeVideo(cue.url))
  })
  // Items dropped from the queue never play their clips; don't pile them up.
  while (clipPool.size > 24) {
    const [key, v] = clipPool.entries().next().value
    clipPool.delete(key)
    v.remove()
  }
}

function retireClip (c) {
  c.video.classList.remove('on')
  setTimeout(() => { c.video.pause(); c.video.remove() }, 600)
}

const FADE_MS = 250 // matches .clip transition; boundary frames match, the fade hides tiny drift

function playClipCue (cue, key) {
  const v = clipPool.get(key) || makeVideo(cue.url)
  clipPool.delete(key)
  v.loop = Boolean(cue.loop)
  v.currentTime = 0
  v.playbackRate = v.defaultPlaybackRate = cue.rate || 1
  v.play().catch(() => {})
  v.classList.add('on')
  // The new take covers the previous one (or the idle loop); tidy up under it.
  const previous = activeClips
  activeClips = [{ video: v, until: now() + cue.dur }]
  setTimeout(() => { previous.forEach(retireClip); parkIdle() }, FADE_MS + 50)
}

function updateClip () {
  const t = now()
  const done = activeClips.filter((c) => t > c.until)
  if (done.length) {
    activeClips = activeClips.filter((c) => t <= c.until)
    done.forEach(retireClip)
    if (!activeClips.length) resumeIdle()
  }
}

// ── Idle continuity ───────────────────────────────────────────────────────
// Every clip starts and ends on the idle loop's first frame. So: while a clip
// covers the idle, park the idle on frame 0; when the clip ends, the idle
// carries on from frame 0; and before the next clip, nudge the idle's speed
// so it is back on frame 0 exactly when that clip starts.
const idleVideo = () => (baseClip.classList.contains('on') && baseClip.duration > 0 ? baseClip : null)
// A *_loop idle is a forward take plus the same take reversed, so it passes
// the first frame twice per loop: at the start and halfway through.
const idlePeriod = (v) => v.duration / (v.currentSrc.includes('_loop.') ? 2 : 1)

function parkIdle () {
  const v = idleVideo()
  if (!v || !activeClips.length) return
  v.pause()
  v.currentTime = 0
  v.playbackRate = 1
}

function bestRate (distance, time) {
  // Speed that covers `distance` seconds of video in `time` seconds, as close to 1 as possible.
  let best = null
  const v = idleVideo()
  for (let k = 0; k < 6; k++) {
    const r = (distance + k * (v ? idlePeriod(v) : 0)) / time
    if (r >= 0.8 && r <= 1.25 && (!best || Math.abs(r - 1) < Math.abs(best - 1))) best = r
  }
  return best
}

function planIdleStart (item) {
  const v = idleVideo()
  const first = item.cues.find((c) => c.kind === 'clip')
  if (!v || !first) return 0.05
  // Flush a clip that just ended first, so nothing resets the idle after planning.
  updateClip()
  const P = idlePeriod(v)
  const c = v.currentTime % P
  if ((c < 0.25 || P - c < 0.1) && first.t < 1.5) {
    // Right on the first frame already: hold it until the clip starts.
    v.pause()
    v.currentTime = Math.round(v.currentTime / P) * P % v.duration
    return 0.05
  }
  let best = null
  for (let k = 0; k < 4; k++) {
    for (const r of [1, 0.92, 1.08, 0.85, 1.15, 1.25]) {
      const delay = (P - c + k * P) / r - first.t
      if (delay < 0.05 || delay > 6) continue
      const score = delay + Math.abs(r - 1) * 4
      if (!best || score < best.score) best = { delay, r, score }
    }
  }
  if (!best) return 0.05
  v.playbackRate = best.r
  return best.delay
}

function resumeIdle () {
  const v = idleVideo()
  if (!v) return
  v.currentTime = 0
  let next = Infinity
  if (playing) {
    const t = now() - playing.start
    const c = playing.item.cues.find((q, i) => q.kind === 'clip' && !playing.fired.has(i) && q.t > t)
    if (c) next = c.t - t
  }
  if (next < 1.6) {
    v.pause() // a short breath between two takes: hold the first frame
    return
  }
  const r = Number.isFinite(next) ? bestRate(0, next) : 1
  v.playbackRate = r || 1
  v.play().catch(() => {})
}
const hasFootage = () => activeClips.length > 0 || baseClip.classList.contains('on') || baseImg.classList.contains('on')
const clipCovering = hasFootage

// ── Drawing: background, felt, hands ──────────────────────────────────────
const FELT = { x: 160, y: 900, w: 760, h: 560, r: 70 }
const HAND_SCALE = 1.3

function makeGrain () {
  const c = document.createElement('canvas')
  c.width = c.height = 256
  const g = c.getContext('2d')
  const img = g.createImageData(256, 256)
  for (let i = 0; i < img.data.length; i += 4) {
    const v = Math.random() * 255
    img.data[i] = img.data[i + 1] = img.data[i + 2] = v
    img.data[i + 3] = 22
  }
  g.putImageData(img, 0, 0)
  return c
}

function makeFelt () {
  const c = document.createElement('canvas')
  c.width = FELT.w + 120
  c.height = FELT.h + 120
  const g = c.getContext('2d')
  g.fillStyle = '#56706d'
  g.fillRect(0, 0, c.width, c.height)
  // Thousands of short wool fibers, light and dark.
  for (let i = 0; i < 14000; i++) {
    const x = Math.random() * c.width
    const y = Math.random() * c.height
    const a = Math.random() * Math.PI * 2
    const len = 3 + Math.random() * 9
    g.strokeStyle = Math.random() < 0.5 ? 'rgba(200,225,215,0.10)' : 'rgba(20,35,35,0.14)'
    g.lineWidth = 0.6 + Math.random() * 0.9
    g.beginPath()
    g.moveTo(x, y)
    g.quadraticCurveTo(x + Math.cos(a + 0.6) * len * 0.5, y + Math.sin(a + 0.6) * len * 0.5,
      x + Math.cos(a) * len, y + Math.sin(a) * len)
    g.stroke()
  }
  return c
}

const grain = makeGrain()
const feltTex = makeFelt()
const dents = [] // { x, y, start }

function roundRect (g, x, y, w, h, r) {
  g.beginPath()
  g.moveTo(x + r, y)
  g.arcTo(x + w, y, x + w, y + h, r)
  g.arcTo(x + w, y + h, x, y + h, r)
  g.arcTo(x, y + h, x, y, r)
  g.arcTo(x, y, x + w, y, r)
  g.closePath()
}

function capsule (g, x, y, w, len, angle) {
  g.save()
  g.translate(x, y)
  g.rotate(angle)
  roundRect(g, -w / 2, -len, w, len + w / 2, w / 2)
  g.fill()
  g.restore()
}

// A stylised hand seen from above, fingers pointing up (-y), wrist at (0,0).
function handShape (g) {
  roundRect(g, -78, -150, 156, 190, 70)
  g.fill()
  capsule(g, -54, -138, 31, 118, -0.10)
  capsule(g, -19, -150, 33, 138, -0.03)
  capsule(g, 17, -148, 32, 130, 0.04)
  capsule(g, 50, -132, 28, 104, 0.12)
  capsule(g, 74, -50, 34, 96, 0.95)
}

function drawHand (g, x, y, rot, mirror, lift) {
  // Shadow: further away and softer the higher the hand is lifted.
  g.save()
  g.translate(x + 10 + lift * 34, y + 22 + lift * 60)
  g.rotate(rot)
  g.scale((mirror ? -1 : 1) * HAND_SCALE, HAND_SCALE)
  g.filter = `blur(${10 + lift * 18}px)`
  g.fillStyle = `rgba(0,0,0,${0.42 - lift * 0.18})`
  handShape(g)
  g.restore()

  g.save()
  g.translate(x, y)
  g.rotate(rot)
  const s = HAND_SCALE * (1 + lift * 0.07) // closer to the camera when lifted
  g.scale((mirror ? -1 : 1) * s, s)
  const grad = g.createLinearGradient(0, -280, 0, 60)
  grad.addColorStop(0, '#f2d3c1')
  grad.addColorStop(1, '#c99580')
  g.fillStyle = grad
  handShape(g)
  // Soft highlight along the knuckles.
  g.globalCompositeOperation = 'soft-light'
  const hl = g.createRadialGradient(-10, -150, 10, -10, -150, 150)
  hl.addColorStop(0, 'rgba(255,255,255,0.45)')
  hl.addColorStop(1, 'rgba(255,255,255,0)')
  g.fillStyle = hl
  handShape(g)
  g.restore()
}

function drawBrush (g, x, y, angle) {
  g.save()
  g.translate(x, y)
  g.rotate(angle)
  g.fillStyle = 'rgba(0,0,0,0.35)'
  g.filter = 'blur(12px)'
  roundRect(g, -14 + 18, 30, 28, 300, 14)
  g.fill()
  g.filter = 'none'
  const wood = g.createLinearGradient(-14, 0, 14, 0)
  wood.addColorStop(0, '#6b4630')
  wood.addColorStop(1, '#a0704d')
  g.fillStyle = wood
  roundRect(g, -14, 20, 28, 300, 14)
  g.fill()
  g.fillStyle = '#d8c7a6'
  g.beginPath()
  g.moveTo(-16, 24)
  g.quadraticCurveTo(-70, -60, -40, -120)
  g.quadraticCurveTo(0, -150, 40, -120)
  g.quadraticCurveTo(70, -60, 16, 24)
  g.closePath()
  g.fill()
  g.restore()
}

// Hand poses: where each hand sits and how lifted it is, from active actions.
function handPoses (t, active) {
  const float = (p) => Math.sin(t * 0.9 + p) * 6
  const pose = {
    L: { x: 290 + float(0), y: 1500 + float(1.3), rot: 0.2, lift: 0.55 },
    R: { x: 790 + float(2.1), y: 1500 + float(0.4), rot: -0.2, lift: 0.55 },
  }
  for (const a of active) {
    const k = clamp((t - a.start) / a.dur, 0, 1)
    if (a.name === 'tap') {
      // Quick press down onto the felt, then lift back off.
      const local = (t - a.start) / 0.12
      const press = local < 0.35 ? ease(local / 0.35) : 1 - ease((local - 0.35) / 0.65)
      const p = pose[a.side === 'R' ? 'R' : 'L']
      p.lift = Math.min(p.lift, 0.55 - press * 0.55)
      p.y -= press * 26
      if (press > 0.9 && !a.dented) {
        a.dented = true
        dents.push({ x: p.x + (a.side === 'R' ? -26 : 26), y: p.y - 360, start: t })
      }
    } else if (a.name === 'felt') {
      // Both hands rub the felt in alternating circles.
      const fade = Math.min(1, k * 6, (1 - k) * 6)
      const w = t * Math.PI * 2 * 1.45
      pose.L.x += Math.cos(w) * 46 * fade
      pose.L.y += Math.sin(w) * 20 * fade - 40 * fade
      pose.R.x += Math.cos(w + Math.PI) * 46 * fade
      pose.R.y += Math.sin(w + Math.PI) * 20 * fade - 40 * fade
      pose.L.lift = pose.R.lift = 0.55 - 0.5 * fade
    } else if (a.name === 'whisper') {
      // The hand on the voice's side drifts up a little, like leaning in.
      const fade = Math.min(1, k * 5, (1 - k) * 5) * 0.6
      if (a.side === 'L' || a.side === 'C') { pose.L.y -= 30 * fade; pose.L.lift += 0.2 * fade }
      if (a.side === 'R' || a.side === 'C') { pose.R.y -= 30 * fade; pose.R.lift += 0.2 * fade }
    }
  }
  return pose
}

function drawScene (t, active) {
  const g = scene
  g.clearRect(0, 0, W, H)
  // Dim warm room.
  const bg = g.createRadialGradient(W / 2, H * 0.55, 80, W / 2, H * 0.55, H * 0.75)
  bg.addColorStop(0, '#2a1d1b')
  bg.addColorStop(1, '#070506')
  g.fillStyle = bg
  g.fillRect(0, 0, W, H)

  if (!clipCovering()) {
    // Felt pad, nudged by rubbing.
    let shiftX = 0
    let shiftY = 0
    for (const a of active) {
      if (a.name === 'felt') {
        const w = t * Math.PI * 2 * 1.45
        shiftX += Math.cos(w) * 6
        shiftY += Math.sin(w) * 3
      }
    }
    g.save()
    g.shadowColor = 'rgba(0,0,0,0.6)'
    g.shadowBlur = 60
    g.shadowOffsetY = 30
    roundRect(g, FELT.x, FELT.y, FELT.w, FELT.h, FELT.r)
    g.fillStyle = '#4c6361'
    g.fill()
    g.restore()
    g.save()
    roundRect(g, FELT.x, FELT.y, FELT.w, FELT.h, FELT.r)
    g.clip()
    g.drawImage(feltTex, FELT.x - 60 + shiftX, FELT.y - 60 + shiftY)
    // Dents where fingertips pressed, fading out with a ripple.
    for (let i = dents.length - 1; i >= 0; i--) {
      const d = dents[i]
      const age = t - d.start
      if (age > 1.2) { dents.splice(i, 1); continue }
      const a = 1 - age / 1.2
      const rg = g.createRadialGradient(d.x, d.y, 2, d.x, d.y, 46 + age * 40)
      rg.addColorStop(0, `rgba(0,0,0,${0.35 * a})`)
      rg.addColorStop(1, 'rgba(0,0,0,0)')
      g.fillStyle = rg
      g.fillRect(d.x - 120, d.y - 120, 240, 240)
    }
    // Edge vignette so the pad reads as soft and thick.
    const vg = g.createRadialGradient(W / 2, FELT.y + FELT.h / 2, FELT.h * 0.3, W / 2, FELT.y + FELT.h / 2, FELT.w * 0.75)
    vg.addColorStop(0, 'rgba(0,0,0,0)')
    vg.addColorStop(1, 'rgba(0,0,0,0.45)')
    g.fillStyle = vg
    g.fillRect(FELT.x, FELT.y, FELT.w, FELT.h)
    g.restore()

    const brush = active.find((a) => a.name === 'brush')
    const pose = handPoses(t, active)
    drawHand(g, pose.L.x, pose.L.y, pose.L.rot, false, clamp(pose.L.lift, 0, 1))
    drawHand(g, pose.R.x, pose.R.y, pose.R.rot, true, clamp(pose.R.lift, 0, 1))
    if (brush) {
      const k = ease((t - brush.start) / brush.dur)
      const fromX = brush.side === 'RL' || brush.side === 'R' ? 860 : 220
      const toX = brush.side === 'LR' || brush.side === 'R' ? 860 : 220
      drawBrush(g, fromX + (toX - fromX) * k, FELT.y + 210 + Math.sin(k * Math.PI) * 40, (toX > fromX ? 1 : -1) * 0.5)
    }
  }

  g.globalAlpha = 1
  g.drawImage(grain, Math.random() * -60, Math.random() * -60, W + 120, H + 120)
}

// ── Lights ───────────────────────────────────────────────────────────────
// Soft violet / lavender, never warm yellow (what sleep streams use).
const PALETTES = {
  violet: { L: [168, 120, 255], R: [205, 150, 255], C: [190, 170, 255] },
  lavender: { L: [190, 175, 255], R: [215, 190, 255], C: [225, 215, 255] },
  blue: { L: [120, 150, 255], R: [150, 130, 255], C: [170, 190, 255] },
}
const PALETTE = PALETTES[params.get('palette')] || PALETTES.violet
const LIGHT_L = { x: 150, y: 640, color: PALETTE.L }
const LIGHT_R = { x: 930, y: 640, color: PALETTE.R }
const LIGHT_C = { x: 540, y: 980, color: PALETTE.C }
const GUIDE_IN = 4 // seconds breathing in (light rises)
const GUIDE_OUT = 6 // seconds breathing out (light fades)

function drawLight (g, light, intensity) {
  const [r, gg, b] = light.color
  const grad = g.createRadialGradient(light.x, light.y, 0, light.x, light.y, 820)
  grad.addColorStop(0, `rgba(${r},${gg},${b},${0.55 * intensity})`)
  grad.addColorStop(0.35, `rgba(${r},${gg},${b},${0.22 * intensity})`)
  grad.addColorStop(1, `rgba(${r},${gg},${b},0)`)
  g.fillStyle = grad
  g.fillRect(0, 0, W, H)
}

function drawLights (t, active) {
  const g = lightsCtx
  g.clearRect(0, 0, W, H)
  const breath = 0.5 + 0.5 * Math.sin(t * 0.55)
  let l = 0.16 + breath * 0.06 + level.L * 0.85
  let r = 0.16 + (1 - breath) * 0.06 + level.R * 0.85
  const alt = active.find((a) => a.name === 'lights')
  if (alt) {
    // Slow left/right alternation, eased in and out over the cue.
    const k = clamp((t - alt.start) / alt.dur, 0, 1)
    const env = Math.min(1, k * 5, (1 - k) * 5)
    const swing = 0.5 + 0.5 * Math.sin((t - alt.start) * Math.PI * 2 * 0.8)
    l += env * swing * 0.75
    r += env * (1 - swing) * 0.75
  }
  if (active.some((a) => a.name === 'breath')) {
    const k = Math.sin(t * 2.4) * 0.5 + 0.5
    l += 0.12 * k
    r += 0.12 * k
  }
  // Guided breathing: one soft light rises for 4 s (breathe in) and fades
  // for 6 s (breathe out); the side lights dim so the eye settles on it.
  let c = 0
  let phase = ''
  const guide = active.find((a) => a.name === 'guide')
  if (guide) {
    const local = t - guide.start
    const k = clamp(local / guide.dur, 0, 1)
    const env = Math.min(1, k * 8, (1 - k) * 8)
    const cyc = local % (GUIDE_IN + GUIDE_OUT)
    const rise = cyc < GUIDE_IN ? ease(cyc / GUIDE_IN) : 1 - ease((cyc - GUIDE_IN) / GUIDE_OUT)
    phase = cyc < GUIDE_IN ? '吸气' : '呼气'
    c = env * (0.15 + 0.85 * rise)
    l *= 1 - 0.6 * env
    r *= 1 - 0.6 * env
  }
  if (!SHOW_LIGHTS) return
  // Over real footage the overlay is a subtle coloured glow, not the show.
  const k = hasFootage() ? 0.6 : 1
  g.globalCompositeOperation = 'lighter'
  drawLight(g, LIGHT_L, clamp(l * k, 0, 1.2))
  drawLight(g, LIGHT_R, clamp(r * k, 0, 1.2))
  if (c > 0) drawLight(g, LIGHT_C, clamp(c, 0, 1.2))
  g.globalCompositeOperation = 'source-over'
  if (phase && SHOW_CAPTION) {
    g.font = '300 44px "PingFang SC", sans-serif'
    g.textAlign = 'center'
    g.fillStyle = `rgba(240,232,255,${0.25 + 0.5 * c})`
    g.fillText(phase + '…', LIGHT_C.x, 560)
  }
}

// ── Main loop ────────────────────────────────────────────────────────────
let wsState = 'connecting'
function frame () {
  const t = now()
  readLevels()
  fireCues()
  updateClip()
  const active = activeActions(t)
  drawScene(t, active)
  drawLights(t, active)
  if (DEBUG) {
    hud.textContent = `ws ${wsState}  queue ${queue.length}\n` +
      `L ${level.L.toFixed(2)}  R ${level.R.toFixed(2)}\n` +
      `${active.map((a) => `${a.name}:${a.side || ''}`).join(' ') || 'idle'}\n` +
      `${playing ? playing.item.display || playing.item.script : '-'}`
  }
  requestAnimationFrame(frame)
}

// ── Server connection ─────────────────────────────────────────────────────
function connect () {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
  let ping = 0
  ws.onopen = () => { wsState = 'open'; ping = setInterval(() => ws.send('ping'), 20000) }
  ws.onclose = () => { wsState = 'closed'; clearInterval(ping); setTimeout(connect, 2000) }
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.type === 'outfit') loadClips()
    if (msg.type === 'play') {
      // Before the click that unlocks audio, keep only the newest item so the
      // first click doesn't replay a backlog; never fall more than 3 behind.
      if (!ac) queue.length = 0
      queue.push(msg)
      while (queue.length > 3) queue.shift()
      preloadClips(msg)
      playNext()
    }
  }
}

document.getElementById('start').addEventListener('click', () => {
  initAudio()
  document.getElementById('start').classList.add('hidden')
  playNext()
})

loadClips()
setInterval(loadClips, 30000)
connect()
requestAnimationFrame(frame)
