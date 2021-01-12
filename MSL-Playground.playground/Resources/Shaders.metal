
#include <metal_stdlib>
using namespace metal;

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

float distToSphere(Ray ray, Sphere s) {
  return length(ray.origin - s.center) - s.radius;
}

float distToScene(Ray r) {
  Sphere s = Sphere(float3(1.0), 0.5);
  Ray repeatRay = r;
  repeatRay.origin = fmod(r.origin, 2.0);
  return distToSphere(repeatRay, s);
}

float2 getUV(int width, int height, float2 gid) {
  float2 uv = float2(gid) / float2(width, height);
  uv = uv * 2.0 - 1.0;
  return uv;
}

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

kernel void mandel(texture2d<float, access::write> output [[texture(0)]],
                                      constant float &time [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
  
}
