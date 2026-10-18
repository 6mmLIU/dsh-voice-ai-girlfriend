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

// ── Unit playback (gapless) ───────────────────────────────────────────────
// The server streams short units (a take, a whispered line, a transition).
// Each is scheduled to start exactly when the previous one ends, so audio
// and picture run on without gaps; units not started yet can be recalled
// ("flush") when the operator stops or switches an action.
const units = [] // { u, start (stage clock), src, fired:Set, announced }
let tailEnd = 0 // stage time when the last scheduled unit ends
const actions = [] // { name, side, start, dur } for lights / placeholder drawing
const flushed = new Set()
let chain = Promise.resolve()
let ws = null

function send (msg) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg))
}

function onUnit (u) {
  chain = chain.then(() => scheduleUnit(u)).catch((err) => console.error('[asmr] unit failed', err))
}

async function scheduleUnit (u) {
  if (!ac || flushed.has(u.id)) return
  let buffer = null
  if (u.url) buffer = await ac.decodeAudioData(await (await fetch(u.url)).arrayBuffer())
  if (flushed.has(u.id)) return
  const t = now()
  // Back to back with the previous unit; or, coming out of idle, when the
  // idle take reaches its first frame (see planIdleStart).
  const start = tailEnd > t + 0.03 ? tailEnd : t + planIdleStart(u)
  let src = null
  if (buffer) {
    src = ac.createBufferSource()
    src.buffer = buffer
    src.connect(ac.destination)
    src.connect(ac.__splitter)
    src.start(ac.currentTime + Math.max(0, start - now()))
  }
  tailEnd = start + u.duration
  preloadClips(u)
  units.push({ u, start, src, fired: new Set(), announced: false })
}

function flushUnits (ids) {
  ids.forEach((id) => flushed.add(id))
  const t = now()
  for (let i = units.length - 1; i >= 0; i--) {
    const x = units[i]
    if (!ids.includes(x.u.id) || x.start <= t) continue
    try { if (x.src) x.src.stop() } catch {}
    dropPreloaded(x.u)
    units.splice(i, 1)
  }
  tailEnd = units.reduce((m, x) => Math.max(m, x.start + x.u.duration), 0)
}

let captionUntil = 0
function fireCues () {
  const t = now()
  for (const x of units) {
    if (t < x.start) continue
    if (!x.announced) {
      x.announced = true
      send({ type: 'started', id: x.u.id })
    }
    x.u.cues.forEach((cue, i) => {
      if (x.fired.has(i) || x.start + cue.t > t) return
      x.fired.add(i)
      if (cue.kind === 'clip') {
        playClipCue(cue, `${x.u.id}:${i}`)
      } else if (cue.kind === 'speech') {
        actions.push({ name: 'whisper', side: cue.pos, start: x.start + cue.t, dur: cue.dur })
        showCaption(x.u.user, cue.text)
        captionUntil = x.start + cue.t + cue.dur
      } else {
        actions.push({ name: cue.name, side: cue.side, start: x.start + cue.t, dur: cue.dur })
      }
    })
  }
  for (let i = units.length - 1; i >= 0; i--) {
    if (t > units[i].start + units[i].u.duration + 1) units.splice(i, 1)
  }
  if (captionUntil && t > captionUntil + 0.6) {
    hideCaption()
    captionUntil = 0
  }
}

function nextClipIn () {
  // Seconds until the next clip that has not started yet (Infinity if none).
  const t = now()
  let best = Infinity
  for (const x of units) {
    x.u.cues.forEach((c, i) => {
      if (c.kind === 'clip' && !x.fired.has(i)) best = Math.min(best, x.start + c.t - t)
    })
  }
  return best
}

function activeActions (t) {
  for (let i = actions.length - 1; i >= 0; i--) {
    if (t > actions[i].start + actions[i].dur + 1.5) actions.splice(i, 1)
  }
  return actions.filter((a) => t >= a.start && t <= a.start + a.dur)
}

// ── Caption ──────────────────────────────────────────────────────────────
function showCaption (user, text) {
  if (!SHOW_CAPTION || !text) return
  caption.querySelector('.who').textContent = user ? `回复 @${user}` : ''
  caption.querySelector('.what').textContent = text
  caption.classList.add('on')
}
function hideCaption () {
  caption.classList.remove('on')
}

// ── Footage: today's outfit ───────────────────────────────────────────────
let clips = {}
let idleKey = ''
const baseImg = document.getElementById('baseImg')
const baseClip2 = document.getElementById('baseClip2')

async function loadClips () {
  let data = {}
  try {
    data = await (await fetch('/api/media')).json()
  } catch { return }
  clips = data.clips || {}
  // Idle playlist: every calm base-pose take (idle, breathing), never a fixed loop.
  const videos = [...(clips.idle || []), ...(clips.breath || [])].filter((u) => !u.includes('_loop.'))
  const stills = data.stills || []
  const key = JSON.stringify([data.outfit, videos, stills])
  if (key !== idleKey) {
    idleKey = key
    setIdle(videos, stills)
  }
}

const clipPool = new Map() // `${unit}:${cue}` -> preloaded <video>
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

function preloadClips (u) {
  u.cues.forEach((cue, i) => {
    if (cue.kind === 'clip') clipPool.set(`${u.id}:${i}`, makeVideo(cue.url))
  })
  while (clipPool.size > 24) {
    const [key, v] = clipPool.entries().next().value
    clipPool.delete(key)
    v.remove()
  }
}

function dropPreloaded (u) {
  u.cues.forEach((_, i) => {
    const key = `${u.id}:${i}`
    const v = clipPool.get(key)
    if (v) { v.remove(); clipPool.delete(key) }
  })
}

function retireClip (c) {
  c.video.classList.remove('on')
  setTimeout(() => { c.video.pause(); c.video.remove() }, 600)
}

const FADE_MS = 250 // matches .clip transition

function playClipCue (cue, key) {
  const v = clipPool.get(key) || makeVideo(cue.url)
  clipPool.delete(key)
  v.loop = false
  v.currentTime = 0
  v.playbackRate = v.defaultPlaybackRate = cue.rate || 1
  v.play().catch(() => {})
  v.classList.add('on')
  // The new take covers the previous one (or the idle); tidy up under it.
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

// ── Idle playlist ─────────────────────────────────────────────────────────
// Two idle players take turns: while one plays a calm take, the other has the
// next random take loaded; at the take's end (always the first frame, pixel
// identical after stabilize_clips) the other one starts instantly. Each take
// plays at a slightly different speed, so the idle never reads as a loop.
// Before an action clip, the idle is steered so a take ends exactly when the
// clip starts; while clips play, the idle waits on its first frame.
const idleEls = [baseClip, baseClip2]
let idleIdx = 0
let idlePool = []
const randRate = () => 0.9 + Math.random() * 0.2
const idleVideo = () => {
  const v = idleEls[idleIdx]
  return v.classList.contains('on') && v.duration > 0 ? v : null
}
const pickIdle = (except) => {
  const options = idlePool.filter((u) => !except || !except.endsWith(u)) 
  const list = options.length ? options : idlePool
  return list[Math.floor(Math.random() * list.length)]
}

function setIdle (videos, stills) {
  idlePool = videos
  if (videos.length) {
    baseImg.classList.remove('on')
    const cur = idleEls[idleIdx]
    const other = idleEls[1 - idleIdx]
    cur.src = pickIdle()
    cur.playbackRate = cur.defaultPlaybackRate = randRate()
    cur.play().catch(() => {})
    cur.classList.add('on')
    other.classList.remove('on')
    other.src = pickIdle(cur.src)
    other.load()
  } else {
    idleEls.forEach((v) => { v.classList.remove('on'); v.pause() })
    if (stills.length) {
      baseImg.src = stills[Math.floor(Math.random() * stills.length)]
      baseImg.classList.add('on')
    } else {
      baseImg.classList.remove('on')
    }
  }
}

function onIdleEnded (ev) {
  const cur = idleEls[idleIdx]
  if (ev.target !== cur || !cur.classList.contains('on')) return
  const next = idleEls[1 - idleIdx]
  const gap = nextClipIn()
  if (activeClips.length || gap < (next.duration || 5) / 1.25) {
    return // a clip is about to cover the idle: hold this (first-)frame until it does
  }
  next.currentTime = 0
  const fill = next.duration / gap
  next.playbackRate = next.defaultPlaybackRate = Number.isFinite(gap) && fill >= 0.8 && fill <= 1.25 ? fill : randRate()
  next.play().catch(() => {})
  next.classList.add('on')
  cur.classList.remove('on')
  cur.pause()
  idleIdx = 1 - idleIdx
  cur.src = pickIdle(next.currentSrc)
  cur.load()
}
idleEls.forEach((v) => v.addEventListener('ended', onIdleEnded))

function parkIdle () {
  const v = idleVideo()
  if (!v || !activeClips.length) return
  v.pause()
  v.currentTime = 0
}

function planIdleStart (u) {
  // Delay before a unit that starts out of idle, so the idle take is exactly
  // on its first frame when the unit's first clip begins.
  const v = idleVideo()
  const first = u.cues.find((c) => c.kind === 'clip')
  if (!v || !first) return 0.05
  updateClip()
  if (v.paused) return Math.max(0.05, 0.05 - first.t) // already holding the first frame
  const rem = v.duration - v.currentTime
  let best = null
  for (const r of [1, 0.95, 1.05, 0.9, 1.1, 0.85, 1.15, 0.8, 1.25]) {
    const delay = rem / r - first.t
    if (delay < 0.05 || delay > 6) continue
    const score = delay * 0.3 + Math.abs(r - 1) * 4
    if (!best || score < best.score) best = { delay, r, score }
  }
  if (!best) return 0.05 // the take ends first; onIdleEnded then holds its first frame
  v.playbackRate = best.r
  return best.delay
}

function resumeIdle () {
  const v = idleVideo()
  if (!v) return
  v.currentTime = 0
  const gap = nextClipIn()
  if (gap < 1.6) {
    v.pause() // a short breath between two units: hold the first frame
    return
  }
  const fill = v.duration / gap
  v.playbackRate = v.defaultPlaybackRate = Number.isFinite(gap) && fill >= 0.8 && fill <= 1.25 ? fill : randRate()
  v.play().catch(() => {})
}
const hasFootage = () => activeClips.length > 0 || idleEls.some((v) => v.classList.contains('on')) || baseImg.classList.contains('on')
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

// ── Lighting ─────────────────────────────────────────────────────────────
// Lighting works on the footage itself instead of painting coloured blobs:
//  * #lights is a soft-light layer: neutral grey changes nothing; a wide,
//    soft wash from the left or right edge lifts that side of the picture,
//    like a lamp on that side being turned up;
//  * exposure (CSS brightness on the video layers) carries the guided
//    breathing: the whole room slowly brightens on the inhale and dims on
//    the exhale while the vignette closes in.
// Everything stays within a few percent unless a light cue is running.
const TINTS = { violet: [228, 212, 255], lavender: [236, 228, 255], blue: [214, 226, 255], warm: [255, 232, 210] }
const TINT = TINTS[params.get('palette')] || TINTS.violet
const GUIDE_IN = 4 // seconds breathing in (room brightens)
const GUIDE_OUT = 6 // seconds breathing out (room dims)
const guideText = document.getElementById('guideText')
let exposure = 1

function setExposure (v) {
  if (Math.abs(v - exposure) < 0.002) return
  exposure = v
  stageEl.style.setProperty('--exposure', v.toFixed(3))
}

function sideWash (g, side, strength) {
  // side: -1 = left, 1 = right. A wide linear wash, no hard shape.
  if (strength <= 0.003) return
  const [r, gg, b] = TINT
  const x0 = side < 0 ? 0 : W
  const grad = g.createLinearGradient(x0, 0, W / 2 - side * W * 0.15, 0)
  grad.addColorStop(0, `rgba(${r},${gg},${b},${clamp(strength, 0, 0.6)})`)
  grad.addColorStop(1, 'rgba(128,128,128,0)')
  g.fillStyle = grad
  g.fillRect(0, 0, W, H)
}

function vignette (g, strength) {
  if (strength <= 0.003) return
  const grad = g.createRadialGradient(W / 2, H * 0.55, H * 0.28, W / 2, H * 0.55, H * 0.78)
  grad.addColorStop(0, 'rgba(128,128,128,0)')
  grad.addColorStop(1, `rgba(20,14,36,${clamp(strength, 0, 0.7)})`)
  g.fillStyle = grad
  g.fillRect(0, 0, W, H)
}

function drawLights (t, active) {
  const g = lightsCtx
  g.globalCompositeOperation = 'source-over'
  g.fillStyle = 'rgb(128,128,128)' // neutral for soft-light
  g.fillRect(0, 0, W, H)
  if (!SHOW_LIGHTS) {
    setExposure(1)
    guideText.classList.remove('on')
    return
  }
  // The side where the sound is gets a touch brighter.
  let wl = level.L * 0.1
  let wr = level.R * 0.1
  let exp = 1
  let vig = 0.12
  const alt = active.find((a) => a.name === 'lights')
  if (alt) {
    // Slow left / right alternation (one side every 4 s), eased in and out.
    const k = clamp((t - alt.start) / alt.dur, 0, 1)
    const env = Math.min(1, k * 5, (1 - k) * 5)
    const swing = 0.5 + 0.5 * Math.sin((t - alt.start) * Math.PI * 2 / 8)
    wl += env * swing * 0.65
    wr += env * (1 - swing) * 0.65
  }
  if (active.some((a) => a.name === 'breath')) exp += 0.03 * (Math.sin(t * 1.6) * 0.5 + 0.5)
  const guide = active.find((a) => a.name === 'guide')
  let phase = ''
  if (guide) {
    const local = t - guide.start
    const k = clamp(local / guide.dur, 0, 1)
    const env = Math.min(1, k * 6, (1 - k) * 6)
    const cyc = local % (GUIDE_IN + GUIDE_OUT)
    const rise = cyc < GUIDE_IN ? ease(cyc / GUIDE_IN) : 1 - ease((cyc - GUIDE_IN) / GUIDE_OUT)
    phase = cyc < GUIDE_IN ? '吸气…' : '呼气…'
    exp += env * (-0.16 + 0.26 * rise) // dims to ~0.84 on the exhale, up to ~1.1 on the inhale
    vig += env * 0.45 * (1 - rise)
    wl *= 1 - env
    wr *= 1 - env
  }
  sideWash(g, -1, wl)
  sideWash(g, 1, wr)
  vignette(g, vig)
  setExposure(exp)
  if (phase && SHOW_CAPTION) {
    guideText.textContent = phase
    guideText.classList.add('on')
  } else {
    guideText.classList.remove('on')
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
    const cur = units.find((x) => t >= x.start && t < x.start + x.u.duration)
    hud.textContent = `ws ${wsState}  units ${units.length}  next clip ${nextClipIn().toFixed(1)}\n` +
      `L ${level.L.toFixed(2)}  R ${level.R.toFixed(2)}\n` +
      `${active.map((a) => `${a.name}:${a.side || ''}`).join(' ') || 'idle'}\n` +
      `${cur ? cur.u.what || '' : '-'}`
  }
  requestAnimationFrame(frame)
}

// ── Server connection ─────────────────────────────────────────────────────
function connect () {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`)
  let ping = 0
  ws.onopen = () => { wsState = 'open'; ping = setInterval(() => ws.send('ping'), 20000) }
  ws.onclose = () => { wsState = 'closed'; clearInterval(ping); setTimeout(connect, 2000) }
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data)
    if (msg.type === 'outfit') loadClips()
    else if (msg.type === 'unit') onUnit(msg)
    else if (msg.type === 'flush') flushUnits(msg.ids || [])
  }
}

document.getElementById('start').addEventListener('click', () => {
  initAudio()
  document.getElementById('start').classList.add('hidden')
})

loadClips()
setInterval(loadClips, 30000)
connect()
requestAnimationFrame(frame)
