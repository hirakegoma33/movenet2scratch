
// movenet2scratch.mjs
// Xcratch/Scratch3 拡張：MoveNet 版 Pose 拡張（単一人物：Lightning/Thunder切替）

const KP_NAMES = [
  'nose','left_eye','right_eye','left_ear','right_ear',
  'left_shoulder','right_shoulder','left_elbow','right_elbow',
  'left_wrist','right_wrist','left_hip','right_hip',
  'left_knee','right_knee','left_ankle','right_ankle'
];

export default class MoveNet2Scratch {
  constructor(runtime){
    this.runtime = runtime;
    this.video = null;
    this.stream = null;
    this.detector = null;
    this.pose = null;
    this.minScore = 0.3;
    this.running = false;
    this.modelType = 'lightning'; // 'lightning' | 'thunder' | 'multipose'
    this._loop = this._loop.bind(this);
  }

  getInfo(){
    return {
      id: 'movenet2scratch',
      name: 'MoveNet2Scratch',
      color1: '#4B8BF4',
      blocks: [
        { opcode: 'start', blockType: 'command', text: 'MoveNetを開始（[model]）', arguments: {
          model: { type: 'string', menu: 'ModelMenu', defaultValue: 'lightning' }
        }},
        { opcode: 'stop', blockType: 'command', text: 'MoveNetを停止' },
        { opcode: 'setMinScore', blockType: 'command', text: '最小スコアを[score]にする', arguments: {
          score: { type: 'number', defaultValue: 0.3 }
        }},
        { opcode: 'getX', blockType: 'reporter', text: '[part] のx', arguments: {
          part: { type: 'string', menu: 'PartMenu', defaultValue: 'nose' }
        }},
        { opcode: 'getY', blockType: 'reporter', text: '[part] のy', arguments: {
          part: { type: 'string', menu: 'PartMenu', defaultValue: 'nose' }
        }},
        { opcode: 'getScore', blockType: 'reporter', text: '[part] のスコア', arguments: {
          part: { type: 'string', menu: 'PartMenu', defaultValue: 'nose' }
        }},
        { opcode: 'hasPose', blockType: 'boolean', text: 'ポーズを検出できた？' }
      ],
      menus: {
        PartMenu: { acceptReporters: true, items: KP_NAMES },
        ModelMenu: { items: ['lightning','thunder'] } // multiposeは実験向け
      }
    };
  }

  async _ensureLibs(){
    if (this._libsLoaded) return;
    // TensorFlow.js & pose-detection を動的ロード（CDN）
    await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-core@4.20.0/dist/tf-core.min.js');
    await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-webgl@4.20.0/dist/tf-backend-webgl.min.js');
    await import('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-converter@4.20.0/dist/tf-converter.min.js');
    this.poseDetection = await import('https://cdn.jsdelivr.net/npm/@tensorflow-models/pose-detection@3.5.0/dist/pose-detection.min.js');
    // バックエンド初期化
    await window.tf.setBackend('webgl');
    await window.tf.ready();
    this._libsLoaded = true;
  }

  async _openCamera(){
    if (this.video) return;
    this.video = document.createElement('video');
    this.video.autoplay = true;
    this.video.playsInline = true;
    this.video.muted = true;
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: 640, height: 480 }});
    this.video.srcObject = this.stream;
    await this.video.play();
  }

  async _createDetector(){
    const pd = this.poseDetection;
    const model = pd.SupportedModels.MoveNet;
    const type = this.modelType === 'thunder'
      ? pd.movenet.modelType.SINGLEPOSE_THUNDER
      : pd.movenet.modelType.SINGLEPOSE_LIGHTNING;
    this.detector = await pd.createDetector(model, {
      modelType: type,
      enableSmoothing: true
    });
  }

  async start({model}){
    try{
      this.modelType = (model === 'thunder') ? 'thunder' : 'lightning';
      await this._ensureLibs();
      await this._openCamera();
      await this._createDetector();
      this.running = true;
      this._loop();
    }catch(e){
      console.error('MoveNet start error:', e);
    }
  }

  stop(){
    this.running = false;
    if (this.detector){ this.detector.dispose(); this.detector = null; }
    if (this.stream){
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.video = null;
    this.pose = null;
  }

  setMinScore({score}){ this.minScore = Math.max(0, Math.min(1, Number(score)||0)); }

  async _loop(){
    if (!this.running || !this.detector || !this.video) return;
    try{
      const poses = await this.detector.estimatePoses(this.video, {flipHorizontal: true}); // ミラー表示想定
      this.pose = (poses && poses[0]) || null;
    }catch(e){ /* 継続 */ }
    // 次フレーム
    if (this.running) window.requestAnimationFrame(this._loop);
  }

  _getKeypoint(part){
    if (!this.pose || !this.pose.keypoints) return null;
    const idx = KP_NAMES.indexOf(String(part));
    if (idx < 0) return null;
    const kp = this.pose.keypoints[idx];
    if (!kp || kp.score < this.minScore) return null;
    return kp; // {x, y, score, name}
  }

  getX({part}){ const kp = this._getKeypoint(part); return kp ? Math.round(kp.x) : 0; }
  getY({part}){ const kp = this._getKeypoint(part); return kp ? Math.round(kp.y) : 0; }
  getScore({part}){ const kp = this._getKeypoint(part); return kp ? Number(kp.score.toFixed(3)) : 0; }
  hasPose(){ return !!(this.pose && this.pose.keypoints && this.pose.keypoints.some(k=>k.score>=this.minScore)); }
}
