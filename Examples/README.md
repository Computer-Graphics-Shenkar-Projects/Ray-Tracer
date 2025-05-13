# Input Format

The ray tracer reads scene descriptions from plain text files. Each scene file consists of lines representing different components of the scene, including camera setup, global lighting, light sources, and geometric objects.

**Scene Settings:**

e (Eye) – Camera position as a 3D coordinate:
Format: e x y z _
The 4th value can be ignored.

a (Ambient) – Global ambient light intensity (r, g, b):
Format: a r g b 1.0
The last value is always 1.0 and can be ignored.

**Light Sources:**

d (Direction) – Direction of the light:
Format: d x y z w

w = 0.0 indicates a directional light.

w = 1.0 indicates a spotlight.

p (Position) – Only for spotlights. Position of the light:
Format: p x y z cutoff
cutoff is the cosine of the spotlight’s cutoff angle.

i (Intensity) – Light source color/intensity:
Format: i r g b 1.0
The last value can be ignored.
The order of i entries matches the order of d.

**Objects:**

o (Object) – Sphere or plane definition:

* Sphere: o x y z r → (x, y, z) is the center, r > 0 is the radius.

* Plane: o a b c d → Plane equation coefficients: ax + by + cz + d = 0, d ≤ 0.

c (Color and Material) – Material properties for each object:

Format: c r g b shininess

* r, g, b are used for both ambient and diffuse materials.

* shininess (n) controls the size of specular highlights.

c entries must appear in the same order as o entries.

**Lighting Model: Phong Illumination**
The ray tracer uses the Phong illumination model to compute the color at each surface point. The equation combines ambient, diffuse, and specular reflections, along with shadows.

**Parameter Mapping:**

| Term        | Description              | Source                    |
| ----------- | ------------------------ | ------------------------- |
| **IE**      | Viewer intensity         | Assumed `(0,0,0)`         |
| **IA**      | Global ambient intensity | From `a`                  |
| **li**      | Light source intensity   | From each `i`             |
| **KA**      | Ambient material color   | From `c`                  |
| **KD**      | Diffuse material color   | From `c`                  |
| **KS**      | Specular color           | Assumed `(0.7, 0.7, 0.7)` |
| **Si**      | Shadow flag (0 or 1)     | 0 = blocked, 1 = visible  |
| **n**       | Shininess                | 4th value of `c`          |
| **Vectors** | Normals, directions      | Calculated from geometry  |

This setup enables realistic rendering of shadows, specular highlights, and diffuse shading based on the physical properties of the scene.
