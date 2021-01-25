#include <metal_stdlib>
using namespace metal;

// MARK: Structs

struct Ray {
  float3 origin;
  float3 direction;
  Ray(float3 o, float3 d) {
    origin = o;
    direction = normalize(d);
  }
};

struct Sphere {
  float3 center;
  float radius;
  Sphere(float3 c, float r) {
    center = c;
    radius = r;
  }
  float distToPoint(float3 point){
    return length(point - center) - radius;
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

struct Plane {
  float3 normal;
  float size;
  Plane(float3 n, float s){
    normal = normalize(n);
    size = s;
  }
  float distToPoint(float3 point) {
    return dot(point, normal) + size;
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
  uv.y *= -1.0;
  return uv;
}

float4x4 createViewMatrix(float3 eyePos, float3 focusPos, float3 up) {
  float3 f = normalize(focusPos - eyePos);
  float3 s = normalize(cross(f, up));
  float3 u = cross(s, up);
  return float4x4(float4(s, 0.0), float4(u, 0.0), float4(-f, 0.0), float4(float3(0.0), 1.0));
}

float4x4 createTranslateMatrix(float3 t) {
  float4x4 m;
  m[0] = float4(1.0, 0.0, 0.0, t.x);
  m[1] = float4(0.0, 1.0, 0.0, t.y);
  m[2] = float4(0.0, 0.0, 1.0, t.z);
  m[3] = float4(0.0, 0.0, 0.0, 1.0);
  return m;
}

float2 hash2(float2 p) {
  return fract(float2(5978.23857, 2915.98275)*sin(float2(
                                                         p.x*832.2388 + p.y*234.9852,
                                                         p.x*921.7381 + p.y*498.2348
                                                         )))*2.-1.;
}

float getPerlinValue(float2 uv, float scale, float offset = 0.0){
  uv *= scale;
  float2 f = fract(uv);
  float2 m = f * f * (3.-f-f);
  float2 p = uv - f;
  
  float n = mix(
                mix(dot(hash2(p + offset + float2(0,0)), f - float2(0,0)),
                    dot(hash2(p + offset + float2(1,0)), f - float2(1,0)), m.x),
                mix(dot(hash2(p + offset + float2(0,1)), f - float2(0,1)),
                    dot(hash2(p + offset + float2(1,1)), f - float2(1,1)), m.x),
                m.y);
  
  return float(0.5 * n + 0.5);
}

float3 getPerlinNormal(float2 uv, float scale, float offsetLayer) {
  float n1 = getPerlinValue(uv, scale, 0.0);
  float n2 = getPerlinValue(uv, scale, offsetLayer);
  float3 p1 = float3(n1);
  p1.r = 0.5;
  float3 p2 = float3(n2);
  p2.g = 0.5;
  float3 normalValue = p1 + p2;
  normalValue.b *= 1.2;
  normalValue /= (normalValue.r + normalValue.g + normalValue.b);
  normalValue /= max(max(normalValue.r, normalValue.g), normalValue.b);
  normalValue = normalValue * 2.0 - 1.0;
  return normalValue;
}

float3 getJetSpectra(float2 uv) {
  // w: [400, 700]
  // x: [0,   1]
  // float x = saturate((w - 400.0)/ 300.0);
  float x = saturate((uv.x + 1.0) / 2.0);
  float3 c;
  
  if (x < 0.25)
    c = float3(0.0, 4.0 * x, 1.0);
  else if (x < 0.35)
    c = float3(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
  else if (x < 0.65)
    c = float3(4.0 * (x - 0.35), 1.0, 0.0);
  else if (x < 0.95)
    c = float3(1.0, 1.0 + 4.0 * (0.65 - x), 0.0);
  else
    c = float3(0.4, 0.4, 0.4);
  
  // Clamp colour components in [0,1]
  return saturate(c);
}

// MARK: Phong calculator
float3 phongLighting(float3 ambColor, float3 specColor, float attenuation, float diffuse, float specular){
  return ambColor + attenuation * (diffuse + specular) * (specColor);
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
      float3 surfacePosition = ray.origin;
      float3 ambient = float3(0.2, 0.2, 0.4);
      float3 norm = normalize(sdfNormalEstimate(&distToBoundingBoxScene, ray));
      float3 lightDirection = normalize(camPos - surfacePosition);
      float diffuse = saturate(dot(norm, lightDirection));
      
      float3 cameraDirection = normalize(camPos - surfacePosition);
      
      float specular = 0.0;
      if (diffuse > 0.0) {
        specular = pow(saturate(dot(cameraDirection, reflect(-lightDirection, norm))), 200);
      }
      float distanceToLight = length(camPos - surfacePosition);
      float attenuation = 1.0 / (1.0 + 0.2 * pow(distanceToLight, 2));
            
      col = ambient + attenuation * (diffuse + specular) * (ambient + 0.4);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  output.write(float4(col, 1.0), gid);
}

// MARK: Iridescent Scene
kernel void iridescentHall(texture2d<float, access::write> output [[texture(0)]],
                       constant float &time [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(1000.0 + (sin(time) / 2.5) + 1.0, 1000.0 + (cos(time) / 2.5) + 1.0, time / 2.0);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; i++) {
    float dist = distToBoundingBoxScene(ray);
    if (dist < 0.001) {
      float3 surfacePosition = ray.origin;
      float3 ambient = float3(0.2, 0.2, 0.2);
      float3 norm = normalize(sdfNormalEstimate(&distToBoundingBoxScene, ray));
      float3 lightDirection = normalize(camPos - surfacePosition);
      float diffuse = saturate(dot(norm, lightDirection));
      
      float3 cameraDirection = normalize(camPos - surfacePosition);
      
      float specular = 0.0;
      if (diffuse > 0.0) {
        specular = pow(saturate(dot(cameraDirection, reflect(-lightDirection, norm))), 2000);
      }
      float distanceToLight = length(camPos - surfacePosition);
      float attenuation = 1.0 / (0.9 + 0.2 * pow(distanceToLight, 2));
      
      float3 colXZ = getPerlinNormal(fmod(ray.origin.xz * 2.0 - 1.0, 1.0), 100.0, 123.45);
      float3 colYZ = getPerlinNormal(fmod(ray.origin.yz * 2.0 - 1.0, 1.0), 100.0, 123.45);
      float3 colXY = getPerlinNormal(fmod(ray.origin.xy * 2.0 - 1.0, 1.0), 100.0, 123.45);
      float3 powNorm = abs(norm);
      powNorm *= pow(powNorm, float3(2.0));
      powNorm /= powNorm.x + powNorm.y + powNorm.z;
      float3 normalMap = colYZ * powNorm.x + colXZ * powNorm.y + colXY * powNorm.z;
      float3 bumpNormal = norm + normalMap;
      
      float NdotL = dot(dot(cameraDirection, reflect(-lightDirection, bumpNormal)), bumpNormal);
      float3 ramp = getJetSpectra(fmod(NdotL, 3.0));
      col = (ambient + ramp * 0.2) + attenuation * (diffuse + specular) * (ambient + 0.4);
      break;
    }
    ray.origin += ray.direction * dist;
  }
  output.write(float4(col, 1.0), gid);
}


Plane shadowPlane() {
  return Plane(float3(0.0, 1.0, 0.0), 0.8);
}

Sphere shadowSphere() {
  return Sphere(float3(0.0, 0.0, 2.0), 1.0);
}

float distToShadowScene(Ray ray) {
  Plane plane = shadowPlane();
  Sphere sphere = shadowSphere();
  float distPlane = plane.distToPoint(ray.origin);
  float distSphere = sphere.distToPoint(ray.origin);
  return sdfUnion(distPlane, distSphere);
}

float3 getShadow(Ray ray) {
  Ray nuRay = ray;
  float3 res = float3(1.0);
  float k = 50.0;
  float t = 0.1;
  for (int i = 0.0; i < 100.0; ++i){
    float dist = shadowSphere().distToPoint(nuRay.origin);
    if (dist < 0.001){
      return float3(0.0);
    }
    res = min(res, k * dist / t);
    nuRay.origin += nuRay.direction * dist;
    t += dist;
  }
  return res;
}

kernel void shadowMap(texture2d<float, access::write> output [[texture(0)]],
                     constant float &time [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
  float2 uv = getUV(output.get_width(), output.get_height(), float2(gid));
  float3 camPos = float3(0.0, 0.0, -1.0);
  float3 lightPos = float3(3.0 * sin(time), 0.0, 0.0);
  Ray ray = Ray(camPos, normalize(float3(uv, 1.0)));
  
  // Plane info
  Plane plane = shadowPlane();
  float3 planeAmbient = float3(0.4);
  float3 planeSpecular = planeAmbient + 0.4;
  
  // Sphere info
  Sphere sphere = shadowSphere();
  float3 sphereAmbient = float3(0.5, 0.3, 0.4);
  float3 sphereSpecular = sphereAmbient + 0.4;

  float3 col = float3(0.0);
  for (int i=0.0; i<100.0; ++i) {
    float3 surfacePos = ray.origin;
    float distPlane = plane.distToPoint(ray.origin);
    float distSphere = sphere.distToPoint(ray.origin);
    float dist = distToShadowScene(ray);
    if (dist < 0.001) {
      col = float3(1.0);
      float3 shadow = getShadow(Ray(lightPos, surfacePos - lightPos));
      
      float3 ambColor = float3(0.4);
      float3 specColor = ambColor + 0.4;
      if (distPlane >= distSphere) {
        ambColor = float3(0.2, 0.05, 0.05);
        specColor = float3(1.0, 0.6, 0.6);
      }
        
      float3 normal = normalize(sdfNormalEstimate(&distToShadowScene, ray));
      float3 lightDirection = normalize(lightPos - surfacePos);
      float diffuse = saturate(dot(normal, lightDirection));
      float3 cameraDirection = normalize(camPos - surfacePos);
      
      float specular = 0.0;
      if (diffuse > 0.0) {
        specular = pow(saturate(dot(cameraDirection, reflect(-lightDirection, normal))), 200);
      }
      float distanceToLight = length(camPos - surfacePos);
      float attenuation = 1.0 / (1.0 + 0.2 * pow(distanceToLight, 2));
      
      col = phongLighting(ambColor, specColor, attenuation, diffuse, specular);
      if (distPlane < distSphere)
        col *= shadow;
      break;
    }
    ray.origin += ray.direction * dist;
  }
  output.write(float4(col, 1.0), gid);
}
