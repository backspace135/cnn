# -*- coding: utf-8 -*-
"""本地教学服务器：只负责把 Trainer 的状态和操作转换成 HTTP。

创建 Flask app 不会读取 MNIST；首次训练/查询才构造 Trainer。这样缺少数据时
网页仍能打开并显示明确的准备步骤，单元测试也不需要下载完整数据。
"""
import os
import pickle
import zipfile
from pathlib import Path

import paths  # noqa: F401  保留项目内 libs/ 的导入优先级
from flask import Flask, jsonify, request, send_from_directory
from train import DATA_FILE, Trainer, hardware_info

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
PORT = int(os.environ.get("PORT", "5000"))
trainer = None  # 兼容现有代码与测试；正常运行时按需构造


class ApiError(Exception):
    def __init__(self, code, message, status=400):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def create_app(injected_trainer=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    instance = injected_trainer

    def get_trainer():
        nonlocal instance
        global trainer
        if injected_trainer is None and trainer is not None:
            return trainer
        if instance is None:
            try:
                instance = Trainer()
                trainer = instance
            except (FileNotFoundError, ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
                raise ApiError("data_unavailable", f"无法读取 MNIST：{exc}。请运行 download_data.py。", 503) from exc
        return instance

    @app.errorhandler(ApiError)
    def handle_api_error(exc):
        return jsonify({"ok": False, "msg": exc.message,
                        "error": {"code": exc.code, "message": exc.message}}), exc.status

    @app.errorhandler(413)
    def too_large(_):
        return jsonify({"ok": False, "msg": "请求体过大。",
                        "error": {"code": "request_too_large", "message": "请求体过大。"}}), 413

    def json_object():
        if not request.is_json:
            raise ApiError("invalid_json", "请求内容必须是 JSON 对象。")
        value = request.get_json(silent=True)
        if not isinstance(value, dict):
            raise ApiError("invalid_json", "请求内容必须是 JSON 对象。")
        return value

    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/api/health")
    def health():
        try:
            t = get_trainer()
            snapshot = t.snapshot()
            return jsonify({"ok": True, "data_available": True,
                            "dataset_sizes": snapshot["dataset_sizes"],
                            "device": str(t.device), "status": snapshot["status"],
                            "hardware": snapshot["hardware"]})
        except ApiError as exc:
            return jsonify({"ok": False, "data_available": False,
                            "message": exc.message, "hardware": hardware_info()}), 503

    @app.get("/api/state")
    def state():
        return jsonify(get_trainer().snapshot())

    @app.post("/api/control")
    def control():
        body = json_object()
        action = body.get("action")
        allowed = {"action", "epochs", "lr", "batch_size", "dropout", "device_mode"}
        if set(body) - allowed:
            raise ApiError("invalid_field", "请求包含未知字段。")
        hyper = {k: body[k] for k in allowed - {"action"} if k in body}
        if any(value is None for value in hyper.values()):
            raise ApiError("invalid_parameter", "训练参数不能为 null。")
        if action not in {"start", "pause", "resume", "stop", "reset", "configure"}:
            raise ApiError("invalid_action", f"未知 action: {action}")
        if hyper and action not in {"reset", "configure"}:
            raise ApiError("invalid_field", "只有 reset/configure 可以指定训练参数。")
        t = get_trainer()
        try:
            with t.control_lock:
                if action == "start":
                    t.start()
                elif action == "pause":
                    t.pause()
                elif action == "resume":
                    t.resume()
                elif action == "stop":
                    t.stop()
                elif action == "reset":
                    t.reset(**hyper)
                    t.start()  # 保持原页面的“应用参数并重训”行为
                else:
                    t.configure(**hyper)
        except (ValueError, TypeError) as exc:
            raise ApiError("invalid_parameter", str(exc)) from exc
        except RuntimeError as exc:
            raise ApiError("invalid_state", str(exc), 409) from exc
        snapshot = t.snapshot()
        return jsonify({"ok": True, "status": snapshot["status"],
                        "generation": snapshot["generation"]})

    @app.get("/api/model-info")
    def model_info():
        return jsonify({"ok": True, **get_trainer().model_info()})

    @app.get("/api/experiments")
    def experiments():
        return jsonify({"ok": True, "experiments": get_trainer().experiments()})

    @app.post("/api/predict")
    def predict():
        body = json_object()
        if set(body) != {"pixels"}:
            raise ApiError("invalid_pixels", "请提供 pixels 28×28 数字矩阵。")
        try:
            result = get_trainer().predict(body["pixels"])
        except ValueError as exc:
            raise ApiError("invalid_pixels", str(exc)) from exc
        return jsonify({"ok": True, **result})

    @app.post("/api/model")
    def model_file():
        body = json_object()
        if set(body) != {"action"} or body["action"] not in ("save", "load"):
            raise ApiError("invalid_action", "action 必须为 save 或 load。")
        t = get_trainer()
        try:
            path = t.save_model() if body["action"] == "save" else t.load_model()
        except FileNotFoundError as exc:
            raise ApiError("model_missing", str(exc), 404) from exc
        except (ValueError, TypeError, KeyError, OSError, EOFError, pickle.UnpicklingError) as exc:
            raise ApiError("invalid_model", "本地模型文件损坏或版本不兼容。") from exc
        except RuntimeError as exc:
            raise ApiError("invalid_state", str(exc), 409) from exc
        return jsonify({"ok": True, "model": Path(path).name, "status": t.snapshot()["status"]})

    return app


app = create_app()

if __name__ == "__main__":
    print(f"CNN 入门课：http://127.0.0.1:{PORT}")
    print("浏览器中点击“开始训练”；可用 AUTO_TRAIN=1 启动演示自动训练。")
    if os.environ.get("AUTO_TRAIN") == "1":
        with app.test_client() as client:
            response = client.post("/api/control", json={"action": "start"})
            if response.status_code != 200:
                print("自动训练未启动：", response.get_json().get("msg"))
    app.run(host="127.0.0.1", port=PORT, threaded=True, debug=False)
