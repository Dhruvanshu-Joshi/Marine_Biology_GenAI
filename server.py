# import traceback
# import os
# from fastapi import FastAPI, Query
# from fastapi.responses import JSONResponse, FileResponse
# import uvicorn
# from generate_3d import generate_3d

# app = FastAPI(title="3D Asset Generator API")

# @app.get("/")
# def root():
#     return {"message": "3D Asset Generator API is running."}

# @app.post("/generate")
# def generate(
#     concept: str = Query(..., description="Name of the object/creature (e.g., 'Grimpoteuthis')"),
#     method: str = Query("genai", description="Image generation method: 'genai' or 'fathomnet'")
# ):
#     try:
#         results = generate_3d(concept=concept, method=method)
#         return JSONResponse(content={"status": "success", "results": results})
#     except Exception as e:
#         tb = traceback.format_exc()
#         print("Exception in /generate:", tb)   # log full traceback to server console
#         return JSONResponse(
#             content={"status": "error", "message": str(e), "traceback": tb},
#             status_code=500
#         )

# @app.get("/files")
# def get_file(path: str):
#     if not os.path.exists(path):
#         return JSONResponse(content={"error": "File not found"}, status_code=404)
#     return FileResponse(path, filename=os.path.basename(path))

# if __name__ == "__main__":
#     uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=True)

# server.py
# from flask import Flask, request, jsonify
# import os, uuid
# from threading import Thread
# from generate_3d import generate_3d  # import your actual function

# app = Flask(__name__)
# OUTPUT_FOLDER = "output_assets"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# jobs = {}

# def process_job(job_id, concept, method):
#     try:
#         result_paths = generate_3d(concept=concept, method=method, output_dir=OUTPUT_FOLDER)
#         jobs[job_id]["status"] = "done"
#         jobs[job_id]["results"] = result_paths
#     except Exception as e:
#         jobs[job_id]["status"] = "error"
#         jobs[job_id]["error"] = str(e)

# @app.route('/start_job', methods=['POST'])
# def start_job():
#     data = request.get_json(force=True)
#     concept = data.get("concept")
#     method = data.get("method", "genai")

#     if not concept:
#         return jsonify({"error": "Missing 'concept' field"}), 400

#     job_id = str(uuid.uuid4())
#     jobs[job_id] = {"status": "processing"}

#     # Run in background thread
#     Thread(target=process_job, args=(job_id, concept, method)).start()

#     return jsonify({"job_id": job_id}), 202

# @app.route('/job_status/<job_id>', methods=['GET'])
# def job_status(job_id):
#     if job_id not in jobs:
#         return jsonify({"error": "Job ID not found"}), 404
#     return jsonify(jobs[job_id]), 200

# if __name__ == "__main__":
#     # app.run(port=5000, debug=True)
#     app.run(host="0.0.0.0", port=5000, debug=True)


from flask import Flask, request, jsonify
import os
import torch
from generate_3d import generate_3d  # this should expect models passed in

# -------------------------------
# Load models globally at startup
# -------------------------------
print("[INFO] Loading models once at startup...")

# if your generate_3d internally loads models, move that logic here
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

DEVICE = torch.device(
    "cuda:1" if torch.cuda.device_count() > 1
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)

MESH_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'models',
    device=DEVICE
)

PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(
    'models',
    subfolder='hunyuan3d-paint-v2-0-turbo',
)

print("[INFO] Models loaded successfully.")

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    concept = data.get("concept")
    method = data.get("method", "genai")

    if not concept:
        return jsonify({"error": "Missing 'concept' field"}), 400

    try:
        result_paths = generate_3d(
            concept=concept,
            method=method,
            output_dir="output_assets"
        )
        return jsonify({"status": "done", "results": result_paths}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    # Disable reloader to avoid duplicate processes
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
