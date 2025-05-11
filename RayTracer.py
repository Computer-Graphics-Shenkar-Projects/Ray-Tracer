import numpy as np
import math
from PIL import Image
import os
import sys

# ---------- Vector Utilities ----------
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

# ---------- Scene Class (✅ Added This Class) ----------
class Scene:
    def __init__(self):
        self.objects = []         # ✅ Added list to hold all objects (spheres/planes)
        self.lights = []          # ✅ Added list to hold directional lights
        self.spotlights = []      # ✅ Added list to hold spotlight lights
        self.ambient = np.array([0.0, 0.0, 0.0])  # ✅ Default ambient light

# ---------- Material and Object Classes ----------
class Material:
    def __init__(self, ambient, diffuse, specular, shininess):
        self.ambient = np.array(ambient)
        self.diffuse = np.array(diffuse)
        self.specular = np.array(specular)
        self.shininess = shininess

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        L = self.center - ray_origin
        tca = np.dot(L, ray_dir)
        d2 = np.dot(L, L) - tca * tca
        if d2 > self.radius ** 2:
            return None
        thc = math.sqrt(self.radius ** 2 - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > 1e-4:
            return t0
        if t1 > 1e-4:
            return t1
        return None

    def normal_at(self, point):
        return normalize(point - self.center)

class Plane:
    def __init__(self, normal, d, material):
        self.normal = normalize(np.array(normal))
        self.d = d
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        denom = np.dot(self.normal, ray_dir)
        if abs(denom) < 1e-6:
            return None
        t = -(np.dot(self.normal, ray_origin) + self.d) / denom
        return t if t > 1e-4 else None

    def normal_at(self, point):
        return self.normal

# ---------- Light Classes ----------
class DirectionalLight:
    def __init__(self, direction, intensity):
        self.direction = normalize(np.array(direction))
        self.intensity = np.array(intensity)

class Spotlight:
    def __init__(self, position, direction, intensity, cutoff):
        self.position = np.array(position)
        self.direction = normalize(np.array(direction))
        self.intensity = np.array(intensity)
        self.cutoff = cutoff  # cosine of cutoff angle

# ---------- Load Scene from File ----------
def load_scene(path):
    print(f"Loading scene from {path}...")
    if not os.path.exists(path):
        print(f"❌ Scene file '{path}' not found.")
        sys.exit(1)

    scene = Scene()  # ✅ Using the Scene class
    camera_pos = np.array([0, 0, 4], dtype=float)
    objects = []
    materials = []

    dir_lights = []
    spot_lights = []
    spot_dirs = []
    spot_cutoffs = []

    with open(path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts: continue

        tag, *vals = parts
        vals = list(map(float, vals))

        if tag == 'e':
            camera_pos = np.array(vals[:3])
            print("Camera position:", camera_pos)
        elif tag == 'a':
            scene.ambient = np.array(vals[:3])
            print("Ambient light:", scene.ambient)
        elif tag == 'o':
            if vals[3] > 0:  # sphere
                obj = Sphere(np.array(vals[:3]), vals[3], None)
            else:  # plane
                obj = Plane(np.array(vals[:3]), vals[3], None)
            objects.append(obj)
        elif tag == 'c':
            color = np.array(vals[:3])
            shininess = vals[3]
            mat = Material(color, color, np.array([0.7, 0.7, 0.7]), shininess)
            materials.append(mat)
        elif tag == 'd':
            spot_dirs.append(np.array(vals[:3]))
        elif tag == 'p':
            spot_lights.append(np.array(vals[:3]))
            spot_cutoffs.append(vals[3])
        elif tag == 'i':
            dir_lights.append(np.array(vals[:3]))

    # ✅ Assign materials to objects
    if len(materials) != len(objects):
        print("⚠ Warning: Number of materials doesn't match number of objects.")
    for i, obj in enumerate(objects):
        if i < len(materials):
            obj.material = materials[i]
        else:
            obj.material = Material(np.array([0.1, 0.1, 0.1]), np.array([0.1, 0.1, 0.1]), np.array([0.7, 0.7, 0.7]), 8)
        scene.objects.append(obj)

    # ✅ Attach lights
    for i, color in enumerate(dir_lights):
        if i < len(spot_dirs) and i < len(spot_lights):
            dir = spot_dirs[i]
            pos = spot_lights[i]
            cutoff = spot_cutoffs[i]
            scene.spotlights.append(Spotlight(pos, dir, color, cutoff))  # ✅ Use scene.spotlights
            print(f"Added spotlight: pos={pos}, dir={dir}, cutoff={cutoff}")
        else:
            scene.lights.append(DirectionalLight(spot_dirs[i], color))  # ✅ Use scene.lights
            print(f"Added directional light: dir={spot_dirs[i]}, color={color}")

    print("Scene successfully loaded.")
    return scene, camera_pos

# ---------- Phong Lighting ----------
def phong_lighting(point, normal, view_dir, material, lights, spotlights, objects, ambient_light):
    color = material.ambient * ambient_light

    for light in lights:
        L = -light.direction
        shadow_ray_origin = point + normal * 1e-4
        in_shadow = any(
            obj.intersect(shadow_ray_origin, L) is not None for obj in objects
        )
        if in_shadow:
            continue
        NdotL = max(np.dot(normal, L), 0)
        R = normalize(2 * NdotL * normal - L)
        spec = max(np.dot(view_dir, R), 0) ** material.shininess
        color += material.diffuse * light.intensity * NdotL + material.specular * light.intensity * spec

    for s in spotlights:
        Lvec = s.position - point
        L = normalize(Lvec)
        spot_dir = normalize(s.direction)
        spot_cos = np.dot(-L, spot_dir)
        if spot_cos < s.cutoff:
            continue
        shadow_ray_origin = point + normal * 1e-4
        dist_to_light = np.linalg.norm(Lvec)
        in_shadow = any(
            (t := obj.intersect(shadow_ray_origin, L)) is not None and t < dist_to_light
            for obj in objects
        )
        if in_shadow:
            continue
        NdotL = max(np.dot(normal, L), 0)
        R = normalize(2 * NdotL * normal - L)
        spec = max(np.dot(view_dir, R), 0) ** material.shininess
        attenuation = spot_cos
        color += (material.diffuse * s.intensity * NdotL + material.specular * s.intensity * spec) * attenuation

    return np.clip(color, 0, 1)

# ---------- Rendering ----------
def render(eye, objects, lights, spotlights, ambient_light, width=800, height=800):
    print("Starting render...")
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for y in range(height):
        if y % 100 == 0:
            print(f"Rendering row {y}/{height}")
        for x in range(width):
            px = (x + 0.5) / width
            py = (y + 0.5) / height
            screen_x = 2 * px - 1
            screen_y = 1 - 2 * py
            pixel_pos = np.array([screen_x, screen_y, 0])
            ray_dir = normalize(pixel_pos - eye)

            min_t = float('inf')
            hit_obj = None
            for obj in objects:
                t = obj.intersect(eye, ray_dir)
                if t and t < min_t:
                    min_t = t
                    hit_obj = obj

            if hit_obj is not None:
                point = eye + ray_dir * min_t
                normal = hit_obj.normal_at(point)
                view_dir = normalize(eye - point)
                color = phong_lighting(point, normal, view_dir, hit_obj.material, lights, spotlights, objects, ambient_light)
                pixels[x, y] = tuple((color * 255).astype(np.uint8))
            else:
                pixels[x, y] = (0, 0, 0)

    image.save("rendered_scene.png")
    print("✅ Rendered image saved as 'rendered_scene.png'")
    print("✅ Ray Tracer Finished.")

# ---------- Main ----------
def main():
    print("Ray Tracer Started.")
    try:
        scene, eye = load_scene("scene3.txt")  # ✅ fixed scene return
    except FileNotFoundError:
        print("❌ Scene file not found.")
        return
    render(eye, scene.objects, scene.lights, scene.spotlights, scene.ambient)

if __name__ == "__main__":
    main()
