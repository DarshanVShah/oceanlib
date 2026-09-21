// Minimal linear algebra for the viewer.
//
// Written out rather than pulling in GLM: the viewer needs exactly one matrix
// product, one inverse and two matrix constructors, and adding a third-party
// dependency to a demo whose parent library advertises zero dependencies would
// be a poor look. It is also the only place the Vulkan clip-space conventions
// are encoded, so having them in one small readable file is a feature.
#pragma once

#include <cmath>

namespace vkm {

struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3& operator+=(Vec3& a, Vec3 b) { a = a + b; return a; }

inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(Vec3 a, Vec3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline Vec3 normalize(Vec3 v)
{
    const float len = std::sqrt(dot(v, v));
    return (len > 0.0f) ? v * (1.0f / len) : Vec3{0.0f, 1.0f, 0.0f};
}

// Column-major, matching GLSL's `mat4`: m[col][row].
struct Mat4 {
    float m[4][4]{};

    static Mat4 identity()
    {
        Mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i][i] = 1.0f;
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b)
{
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a.m[k][row] * b.m[c][k];
            r.m[c][row] = sum;
        }
    }
    return r;
}

// Rigid-body helpers for placing props. Mat4 is COLUMN-major (m[col][row]),
// matching the multiply above, so a translation lives in column 3.
inline Mat4 translate(Vec3 t)
{
    Mat4 r = Mat4::identity();
    r.m[3][0] = t.x; r.m[3][1] = t.y; r.m[3][2] = t.z;
    return r;
}

inline Mat4 rotate_x(float a)
{
    Mat4 r = Mat4::identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[1][1] = c; r.m[2][1] = -s;
    r.m[1][2] = s; r.m[2][2] =  c;
    return r;
}

inline Mat4 rotate_y(float a)
{
    Mat4 r = Mat4::identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0][0] =  c; r.m[2][0] = s;
    r.m[0][2] = -s; r.m[2][2] = c;
    return r;
}

inline Mat4 rotate_z(float a)
{
    Mat4 r = Mat4::identity();
    const float c = std::cos(a), s = std::sin(a);
    r.m[0][0] = c; r.m[1][0] = -s;
    r.m[0][1] = s; r.m[1][1] =  c;
    return r;
}

// Right-handed look-at producing a view matrix with -Z forward.
inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up)
{
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;  r.m[1][0] = s.y;  r.m[2][0] = s.z;
    r.m[0][1] = u.x;  r.m[1][1] = u.y;  r.m[2][1] = u.z;
    r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
    r.m[3][0] = -dot(s, eye);
    r.m[3][1] = -dot(u, eye);
    r.m[3][2] = dot(f, eye);
    return r;
}

// Vulkan-convention perspective projection.
//
// Two differences from the OpenGL matrix people usually have memorised, and
// both produce silently wrong images rather than errors:
//   - Vulkan's NDC has +Y pointing DOWN, so m[1][1] is negated. Forgetting
//     this renders the scene upside down.
//   - Vulkan's depth range is [0, 1], not [-1, 1], so the third column differs.
//     Using the GL form leaves half the depth buffer unused and makes near
//     geometry z-fight.
inline Mat4 perspective(float fov_y_radians, float aspect, float z_near,
                        float z_far)
{
    const float t = 1.0f / std::tan(fov_y_radians * 0.5f);

    Mat4 r;
    r.m[0][0] = t / aspect;
    r.m[1][1] = -t;  // Y flip for Vulkan
    r.m[2][2] = z_far / (z_near - z_far);
    r.m[2][3] = -1.0f;
    r.m[3][2] = (z_near * z_far) / (z_near - z_far);
    return r;
}

// General 4x4 inverse, used to unproject pixel rays in the sky shader.
// Cofactor expansion; not fast, but it runs once per frame on the CPU.
inline Mat4 inverse(const Mat4& in)
{
    const float* a = &in.m[0][0];
    float inv[16];

    inv[0]  =  a[5]*a[10]*a[15] - a[5]*a[11]*a[14] - a[9]*a[6]*a[15] + a[9]*a[7]*a[14] + a[13]*a[6]*a[11] - a[13]*a[7]*a[10];
    inv[4]  = -a[4]*a[10]*a[15] + a[4]*a[11]*a[14] + a[8]*a[6]*a[15] - a[8]*a[7]*a[14] - a[12]*a[6]*a[11] + a[12]*a[7]*a[10];
    inv[8]  =  a[4]*a[9]*a[15]  - a[4]*a[11]*a[13] - a[8]*a[5]*a[15] + a[8]*a[7]*a[13] + a[12]*a[5]*a[11] - a[12]*a[7]*a[9];
    inv[12] = -a[4]*a[9]*a[14]  + a[4]*a[10]*a[13] + a[8]*a[5]*a[14] - a[8]*a[6]*a[13] - a[12]*a[5]*a[10] + a[12]*a[6]*a[9];
    inv[1]  = -a[1]*a[10]*a[15] + a[1]*a[11]*a[14] + a[9]*a[2]*a[15] - a[9]*a[3]*a[14] - a[13]*a[2]*a[11] + a[13]*a[3]*a[10];
    inv[5]  =  a[0]*a[10]*a[15] - a[0]*a[11]*a[14] - a[8]*a[2]*a[15] + a[8]*a[3]*a[14] + a[12]*a[2]*a[11] - a[12]*a[3]*a[10];
    inv[9]  = -a[0]*a[9]*a[15]  + a[0]*a[11]*a[13] + a[8]*a[1]*a[15] - a[8]*a[3]*a[13] - a[12]*a[1]*a[11] + a[12]*a[3]*a[9];
    inv[13] =  a[0]*a[9]*a[14]  - a[0]*a[10]*a[13] - a[8]*a[1]*a[14] + a[8]*a[2]*a[13] + a[12]*a[1]*a[10] - a[12]*a[2]*a[9];
    inv[2]  =  a[1]*a[6]*a[15]  - a[1]*a[7]*a[14]  - a[5]*a[2]*a[15] + a[5]*a[3]*a[14] + a[13]*a[2]*a[7]  - a[13]*a[3]*a[6];
    inv[6]  = -a[0]*a[6]*a[15]  + a[0]*a[7]*a[14]  + a[4]*a[2]*a[15] - a[4]*a[3]*a[14] - a[12]*a[2]*a[7]  + a[12]*a[3]*a[6];
    inv[10] =  a[0]*a[5]*a[15]  - a[0]*a[7]*a[13]  - a[4]*a[1]*a[15] + a[4]*a[3]*a[13] + a[12]*a[1]*a[7]  - a[12]*a[3]*a[5];
    inv[14] = -a[0]*a[5]*a[14]  + a[0]*a[6]*a[13]  + a[4]*a[1]*a[14] - a[4]*a[2]*a[13] - a[12]*a[1]*a[6]  + a[12]*a[2]*a[5];
    inv[3]  = -a[1]*a[6]*a[11]  + a[1]*a[7]*a[10]  + a[5]*a[2]*a[11] - a[5]*a[3]*a[10] - a[9]*a[2]*a[7]   + a[9]*a[3]*a[6];
    inv[7]  =  a[0]*a[6]*a[11]  - a[0]*a[7]*a[10]  - a[4]*a[2]*a[11] + a[4]*a[3]*a[10] + a[8]*a[2]*a[7]   - a[8]*a[3]*a[6];
    inv[11] = -a[0]*a[5]*a[11]  + a[0]*a[7]*a[9]   + a[4]*a[1]*a[11] - a[4]*a[3]*a[9]  - a[8]*a[1]*a[7]   + a[8]*a[3]*a[5];
    inv[15] =  a[0]*a[5]*a[10]  - a[0]*a[6]*a[9]   - a[4]*a[1]*a[10] + a[4]*a[2]*a[9]  + a[8]*a[1]*a[6]   - a[8]*a[2]*a[5];

    float det = a[0]*inv[0] + a[1]*inv[4] + a[2]*inv[8] + a[3]*inv[12];
    Mat4 out;
    if (det == 0.0f) return Mat4::identity();
    det = 1.0f / det;
    float* o = &out.m[0][0];
    for (int i = 0; i < 16; ++i) o[i] = inv[i] * det;
    return out;
}

}  // namespace vkm
