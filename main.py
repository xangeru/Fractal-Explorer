
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")

app = Flask(__name__)   #flask로 간단하게 파이썬 로컬서버를 열어서 프론트엔드와 연결.
CORS(app)

# 만델브로트 함수
def mandelbrot(width=1200, height=800, max_iter=200,
               x_min=-2.5, x_max=1.0, y_min=-1.25, y_max=1.25,
               cmap="turbo"):

    x = np.linspace(x_min, x_max, width, dtype=np.float64)
    y = np.linspace(y_min, y_max, height, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    div_time = np.zeros(C.shape, dtype=np.int32)

    mask = np.ones(C.shape, dtype=bool)
    for i in range(1, max_iter + 1):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        mask_new = np.abs(Z) <= 2.0
        just_diverged = mask & (~mask_new)
        div_time[just_diverged] = i
        mask = mask_new
        if not mask.any():
            break

    div_time[div_time == 0] = max_iter

    # 이미지 메모리에 저장
    fig = plt.figure(figsize=(10, 6), dpi=150)
    plt.imshow(div_time, extent=[x_min, x_max, y_min, y_max],
               origin="lower", cmap=cmap, interpolation="nearest")
    plt.axis("off")

    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode()
# ----------------------------------------------------


@app.route("/mandelbrot", methods=["POST"])
def generate_mandelbrot():
    data = request.json

    width = int(data.get("width", 1200))
    height = int(data.get("height", 800))
    max_iter = int(data.get("max_iter", 200))
    x_min = float(data.get("x_min", -2.5))
    x_max = float(data.get("x_max", 1.0))
    y_min = float(data.get("y_min", -1.25))
    y_max = float(data.get("y_max", 1.25))
    cmap = data.get("cmap", "turbo")

    img_base64 = mandelbrot(
        width, height, max_iter,
        x_min, x_max,
        y_min, y_max,
        cmap
    )

    return jsonify({"image": img_base64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
