# -*- coding: utf-8 -*-
"""
手写数字识别 —— 训练可视化 & 机器学习学习 Web 服务器 (Flask)

职责只有两件事：
  1. 把 trainer 后台训练产生的实时状态 (self.state) 通过 JSON 暴露出去；
  2. 把浏览器请求转发成"控制训练"的动作（开始/暂停/继续/重置/调参）。

为什么用一个独立 web 层？
  —— 让"训练(后端)"和"展示(前端)"彻底解耦：以后换模型、换数据集、
     甚至换 GPU 服务器，网页面板一行都不用改。

接口：
  GET  /               学习型可视化面板 (static/index.html)
  GET  /api/state      当前训练状态 (JSON)，浏览器每 0.7s 轮询一次
  POST /api/control    {"action": "start"|"pause"|"resume"|"reset"|"configure",
                        ...可选的超参：epochs/lr/batch_size/dropout}
"""
import os
from pathlib import Path

import paths  # noqa: F401  注入 libs/ 里的 torch

from flask import Flask, jsonify, request, send_from_directory
from train import Trainer

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PORT = int(os.environ.get("PORT", "5000"))

app = Flask(__name__)
trainer = Trainer()


# ------------------------------------------------------------------ 学习面板页
@app.route("/")
def index():
    """返回学习型可视化面板页面。"""
    return send_from_directory(STATIC, "index.html")


# ------------------------------------------------------------------ 读取实时状态
@app.route("/api/state")
def api_state():
    """返回训练实时状态快照（浏览器轮询这个接口）。"""
    return jsonify(trainer.snapshot())


# ------------------------------------------------------------------ 控制训练
@app.route("/api/control", methods=["POST"])
def api_control():
    """
    根据请求体里的 action 控制训练：
      start     启动/继续训练
      pause     暂停
      resume    继续
      reset     按给定超参重建模型并从头开始训练
      configure 只更新超参配置，不重建（下次 reset 生效）
    """
    data = request.get_json(silent=True) or {}
    action = data.get("action", "")

    # reset 时允许带上要改的超参（网页"调参实验室"用）
    hyper = {k: data[k] for k in ("epochs", "lr", "batch_size", "dropout")
             if k in data}

    if action == "start":
        trainer.start()
    elif action == "pause":
        trainer.pause()
    elif action == "resume":
        trainer.resume()
    elif action == "reset":
        trainer.reset(**hyper)      # 重建并自动开始
        trainer.start()
    elif action == "configure":
        trainer.configure(**hyper)  # 只改配置，等 reset 生效
    else:
        return jsonify({"ok": False, "msg": f"未知 action: {action}"}), 400
    return jsonify({"ok": True, "status": trainer.snapshot()["status"]})


# ------------------------------------------------------------------ 启动
if __name__ == "__main__":
    print("=" * 64)
    print("  手写数字识别 | 训练可视化 & 机器学习学习面板")
    print(f"  请在浏览器打开:  http://127.0.0.1:{PORT}")
    print("  训练自动开始；页面顶部可控制 开始/暂停/重置/调参")
    print("=" * 64)
    trainer.start()  # 自动开始训练，让面板一打开就能看到实时过程
    # threaded=True 让 /api/state 与训练线程并行，不会互相阻塞
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
