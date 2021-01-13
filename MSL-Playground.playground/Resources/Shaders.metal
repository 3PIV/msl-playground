#include <metal_stdlib>
using namespace metal;

// MARK: Structs

struct Ray {
  float3 origin;
  float3 direction;
  Ray(float3 o, float3 d) {
    origin = o;
    direction = d;
  }
};

struct Sphere {
  float3 center;
  float radius;
  Sphere(float3 c, float r) {
    center = c;
    radius = r;
  }
};

struct Box {
  float3 center;
  float3 bounds;
  Box(float3 c, float3 b) {
    center = c;
    bounds = b;
  }
};

struct BoundingBox {
  Box box;
  float edge;
  BoundingBox(float3 c, float3 b, float e): box(c, b){
    edge = e;
  }
};

// MARK: Distance Functions

float distToSphere(Ray ray, Sphere s) {
  return length(ray.origin - s.center) - s.radius;
}

float distToBox(Ray ray, Box b) {
  float3 q = abs(ray.origin - b.center) - b.bounds;
  return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float distToBoundingBox(Ray ray, BoundingBox bb) {
  float3 p = abs(ray.origin - bb.box.center) - bb.box.bounds;
  float3 q = abs(p + bb.edge) - bb.edge;
  return min(min(
                 length(max(float3(p.x,q.y,q.z),0.0))+min(max(p.x,max(q.y,q.z)),0.0),
                 length(max(float3(q.x,p.y,q.z),0.0))+min(max(q.x,max(p.y,q.z)),0.0)),
             length(max(float3(q.x,q.y,p.z),0.0))+min(max(q.x,max(q.y,p.z)),0.0));
}

// MARK: Intersection Helpers

float3 sdfNormalEstimate(float (*sdfFunc)(Ray), Ray r) {
  float epsilon = 0.01;
  float3 p = r.origin;
  
  float3 estimate = float3(
          sdfFunc(Ray(float3(p.x + epsilon, p.y, p.z), r.direction)) - sdfFunc(Ray(float3(p.x - epsilon, p.y, p.z), r.direction)),
          sdfFunc(Ray(float3(p.x, p.y + epsilon, p.z), r.direction)) - sdfFunc(Ray(float3(p.x, p.y - epsilon, p.z), r.direction)),
          sdfFunc(Ray(float3(p.x, p.y, p.z  + epsilon), r.direction)) - sdfFunc(Ray(float3(p.x, p.y, p.z - epsilon), r.direction))
      );
  return normalize(estimate);
}

float sdfUnion(float d1, float d2) {
  return min(d1, d2);
}

float sdfSubtraction(float d1, float d2) {
  return max(-d1, d2);
}

float sdfIntersection(float d1, float d2) {
  return max(d1, d2);
}

float sdfSmoothUnion( float d1, float d2, float k ) {
  float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) - k * h * (1.0 - h);
}

float sdfSmoothSubtraction( float d1, float d2, float k ) {
  float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
  return mix(d2, -d1, h) + k * h * (1.0 - h);
}

float sdfSmoothIntersection( float d1, float d2, float k ) {
  float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
  return mix(d2, d1, h) + k * h * (1.0 - h);
}

// MARK: Scene Functions

float distToScene(Ray r) {
  Sphere s = Sphere(float3(1.0), 0.5);
  Ray repeatRay = r;
  repeatRay.origin = fmod(r.origin, 2.0);
  return distToSphere(repeatRay, s);
}

float distToBoxScene(Ray r) {
  Box b = Box(float3(1.0), float3(0.5));
  Ray repeatRay = r;
  repeatRay.origin = fmod(r.origin, 2.0);
  return distToBox(repeatRay, b);
}

float distToBoundingBoxScene(Ray r) {
  BoundingBox bb = BoundingBox(float3(1.0), float3(0.3), 0.1);
  Sphere s = Sphere(float(1.0), 0.4);
  Ray repeatRay = r;
  repeatRay.origin = fmod(r.origin, 2.0);
  float d1 = distToBoundingBox(repeatRay, bb);
  float d2 = distToSphere(repeatRay, s);
  float dist = sdfSmoothIntersection(d1, d2, 0.2);
  return dist;
}

float distToMorphingScene(Ray r, float time) {
  BoundingBox bb = BoundingBox(float3(0.0, 0.0, cos(time)), float3(1.0), 0.25);
  //Sphere s2 = Sphere(float3(0.0, sin(time), cos(time)), 0.8);
  Sphere s = Sphere(float3(sin(time), 0.0, 0.0), 0.8);
  Ray repeatRay = r;
  float d1 = distToBoundingBox(repeatRay, bb);
  //float d1 = distToSphere(repeatRay, s2);
  float d2 = distToSphere(repeatRay, s);
  float dist = sdfSmoothIntersection(d2, d1, 0.12);
  return dist;
}

// MARK: Helper Functions

float2 getUV(int width, int height, float2 gid) {
  float2 uv = float2(gid) / float2(width, height);
  uv = uv * 2.0 - 1.0;
  return uv;
}

float4x4 createViewMatrix(float3 eyePos, float3 focusPos, float3 up) {
  float3 f = normalize(focusPos - eyePos);
  float3 s = normalize(cross(f, up));
  float3 u = cross(s, up);
  return float4x4(float4(s, 0.0), float4(u, 0.0), float4(-f, 0.0), float4(float3(0.0), 1.0));
}

// MARK: Sphere Scene
kernel void sphereHall(texture2d<float, access::write> output [[texture(0)]],
                       constant float &time [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(1000.0 + sin(time) + 1.0, 1000.0 + cos(time) + 1.0, time);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; i++) {
    float dist = distToScene(ray);
    if (dist < 0.001) {
      col = float3(1.0);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  float3 posRelativeToCamera = ray.origin - camPos;
  output.write(float4(col * abs((posRelativeToCamera) / 10.0), 1.0), gid);
}

// MARK: Box Scene
kernel void cubeHall(texture2d<float, access::write> output [[texture(0)]],
                     constant float &time [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(1000.0 + sin(time) + 1.0, 1000.0 + cos(time) + 1.0, time);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; i++) {
    float dist = distToBoxScene(ray);
    if (dist < 0.001) {
      col = float3(1.0);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  float3 posRelativeToCamera = ray.origin - camPos;
  output.write(float4(col * abs(posRelativeToCamera / 10.0), 1.0), gid);
}

// MARK: Hollow Scene
kernel void hollowHall(texture2d<float, access::write> output [[texture(0)]],
                       constant float &time [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(1000.0 + (sin(time) / 2.0) + 1.0, 1000.0 + (cos(time) / 2.0) + 1.0, time);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; i++) {
    float dist = distToBoundingBoxScene(ray);
    if (dist < 0.001) {
      col = abs(sdfNormalEstimate(&distToBoundingBoxScene, ray));
      break;
    }
    ray.origin += ray.direction * dist;
  }
  output.write(float4(col, 1.0), gid);
}

kernel void morphing(texture2d<float, access::write> output [[texture(0)]],
                   constant float &time [[buffer(0)]],
                   uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(0.0, 0.0, -1.0);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; i++) {
    float dist = distToMorphingScene(ray, time);
    if (dist < 0.001) {
      //col = float3(1.0 / (smoothstep(0.0, 0.0009, dist) + i));
      float posRelativeToCamera = length(ray.origin - camPos);
      //col = float3(1.0 - (ray.origin + ray.direction * dist).z);
      col = clamp(float3(1.0 - posRelativeToCamera) + 0.3, 0.0, 1.0);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  output.write(float4(col, 1.0), gid);
}
