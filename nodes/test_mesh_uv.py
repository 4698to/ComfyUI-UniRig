import os
import subprocess
import tempfile
import numpy as np
import trimesh

def _save_uv_ppm_debug_varying(mesh: trimesh.Trimesh, out_path: str, size: int = 1024) -> None:
    """Save UV wireframe with support for face-varying UVs."""
    try:
        if not hasattr(mesh, "visual") or mesh.visual is None:
            return
        
        faces = np.asarray(mesh.faces)
        uv = np.asarray(mesh.visual.uv, dtype=np.float64)
        
        n_faces = len(faces)
        n_verts = len(mesh.vertices)
        n_uvs = len(uv)
        
        # 判断 UV 类型
        is_face_varying = (n_uvs == n_faces * 3)
        is_per_vertex = (n_uvs == n_verts)
        
        if not is_face_varying and not is_per_vertex:
            # 尝试 reshape，可能是展平的 face-varying
            if n_uvs % 3 == 0 and n_uvs // 3 == n_faces:
                uv = uv.reshape(-1, 3, 2)
                is_face_varying = True
            else:
                print("UV/Vertex count mismatch: UV=%d, Vert=%d, Face=%d",n_uvs, n_verts, n_faces)
                return
        
        img = np.full((size, size, 3), 255, dtype=np.uint8)
        
        def draw_line(x0, y0, x1, y1):
            # ... 保持原有的 Bresenham 实现 ...
            dx = abs(x1 - x0)
            sx = 1 if x0 < x1 else -1
            dy = -abs(y1 - y0)
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                if 0 <= x0 < size and 0 <= y0 < size:
                    img[y0, x0] = (0, 0, 0)
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy
        
        # 处理 Face-Varying UV (每个三角形的三个角独立UV)
        if is_face_varying:
            print("Face-Varying UV processing...")
            uv_reshaped = uv.reshape(-1, 3, 2) if uv.ndim == 2 else uv
            for face_idx, (face, face_uv) in enumerate(zip(faces, uv_reshaped)):
                # face: [v0, v1, v2], face_uv: [[u0,v0], [u1,v1], [u2,v2]]
                for i in range(3):
                    j = (i + 1) % 3
                    u0, v0 = face_uv[i]
                    u1, v1 = face_uv[j]
                    
                    x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                    y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                    x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                    y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                    draw_line(x0, y0, x1, y1)
        else:
            # 原有的 Per-Vertex UV 处理逻辑
            print("Per-Vertex UV processing...")
            for face in faces:
                n = len(face)
                for i in range(n):
                    a = int(face[i])
                    b = int(face[(i + 1) % n])
                    if a < 0 or b < 0 or a >= len(uv) or b >= len(uv):
                        continue
                    u0, v0 = float(uv[a, 0]), float(uv[a, 1])
                    u1, v1 = float(uv[b, 0]), float(uv[b, 1])
                    x0 = int(max(0, min(size - 1, round(u0 * (size - 1)))))
                    y0 = int(max(0, min(size - 1, round((1.0 - v0) * (size - 1)))))
                    x1 = int(max(0, min(size - 1, round(u1 * (size - 1)))))
                    y1 = int(max(0, min(size - 1, round((1.0 - v1) * (size - 1)))))
                    draw_line(x0, y0, x1, y1)

        with open(out_path, "wb") as f:
            f.write(f"P6\n{size} {size}\n255\n".encode("ascii"))
            f.write(img.tobytes())
        print("Saved UV PPM debug (fixed): %s", out_path)
    except Exception as e:
        print("Failed to save UV PPM debug: %s", e)
        
file_path = r"D:\ComfyUI_xjq\ComfyUI\input\3d\autorig_actor.obj"
loaded = trimesh.load(file_path, process=False, maintain_order=True)
uv_ppm_path = os.path.splitext(file_path)[0] + "_uv_loaded_trimesh.ppm"
_save_uv_ppm_debug_varying(loaded, uv_ppm_path)