# Ray-Tracer

This project is a basic implementation of a ray tracer, a rendering algorithm that simulates the way light interacts with objects to produce realistic images. Ray tracing works by tracing the path of light rays from the observer’s eye (camera) through each pixel of an image plane into a 3D virtual scene. As the rays travel through the scene, they may intersect objects, reflect, refract, or be absorbed, depending on the materials and light sources defined.

The concept of ray tracing differs from traditional rasterization or scanline rendering in that it simulates physical light behavior, such as shadows, reflections, and highlights, which results in visually realistic images. However, this comes at a higher computational cost, making ray tracing more resource-intensive, especially for complex scenes.

**Project Goals and Objectives:**

The main objective of this exercise is to manually implement a basic ray tracer to understand the core principles of ray casting and illumination models. The program performs the following key steps:

  **1.** Shoots rays from the virtual camera through each pixel of the image.

  **2.** Checks for intersections between the ray and scene objects.

  **3.** Determines the closest intersection point to the camera.

  **4.** Calculates the color of the object at that point using the Phong illumination model based on light sources and material properties.

  **5.** Generates a final image by coloring each pixel accordingly.

**Scene Features:**

Display of Geometric Data in 3D Space
* Spheres: Perfectly round 3D objects with defined center and radius, supporting realistic lighting and shading.

* Planes: Infinite flat surfaces that can represent floors, walls, or backdrops in the scene.

* Background: A default color shown when rays do not intersect any object, simulating the environment.

**Light Sources:**

The ray tracer includes several types of basic light sources, each contributing differently to scene illumination:

* Global Ambient Light: A constant, omnipresent light that ensures all surfaces are minimally lit, even without direct illumination.

* Directional Lights: Simulate light coming from a specific direction (e.g., sunlight), providing both diffuse and specular effects.

* Spotlights (optional in advanced versions): Emit light in a cone-shaped region, illuminating only specific areas in the scene.

**Material Properties:**

Each object in the scene has a material defined by its interaction with light:

* Ambient Color: The base color of an object under ambient lighting.

* Diffuse Color: The color under direct lighting, dependent on the angle between the surface and light source.

* Specular Color: The reflective component, producing highlights based on the viewer’s position and surface shininess.

The Phong reflection model is used to combine these components to produce realistic surface appearance.

**Hard Shadows:**

To enhance realism, the ray tracer implements basic hard shadows. This is achieved by casting shadow rays from the surface point to each light source. If the path is blocked by another object, the point is in shadow and receives only ambient light, resulting in sharp, clearly defined shadows.
