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
  float dist = sdfIntersection(d1, d2);
  return dist;
}

// MARK: Helper Functions

float2 getUV(int width, int height, float2 gid) {
  float2 uv = float2(gid) / float2(width, height);
  uv = uv * 2.0 - 1.0;
  return uv;
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
      col = float3(1.0);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  float3 posRelativeToCamera = ray.origin - camPos;
  output.write(float4(col * abs(posRelativeToCamera / 10.0), 1.0), gid);
}

kernel void mandel(texture2d<float, access::write> output [[texture(0)]],
                   constant float &time [[buffer(0)]],
                   uint2 gid [[thread_position_in_grid]]) {
  
}
