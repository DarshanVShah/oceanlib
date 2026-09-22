// Bakes a static (non-animated) glTF/GLB model into a compact, embeddable
// C++ header of flat vertex/index arrays, plus a hull sizing derived from the
// model's own geometry - so the viewer still ships no asset file to find at
// launch and no runtime model loader (ADR-015's reasoning for embedding the
// shaders, applied to a real mesh instead of a procedural one).
//
// This is NOT a general glTF importer. It supports exactly what a Sketchfab-
// exported static prop needs: the GLB container, a node hierarchy built from
// `matrix` or TRS, and POSITION/NORMAL/indices accessors as float32/uint32
// (or uint16/uint8 indices). Skins, animations, sparse accessors and textures
// are not read at all - the boat is drawn flat-shaded, the same as the box
// and rock props, so no image decoding is needed.
//
// Usage: gltf_bake <model.glb> <output.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Minimal JSON (just enough to walk a glTF document: objects, arrays,
// strings, numbers, bools; no unicode escapes beyond passthrough).
// ---------------------------------------------------------------------------
struct Json {
    enum class Kind { Null, Bool, Number, String, Array, Object } kind = Kind::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool has(const std::string& k) const
    {
        return kind == Kind::Object && obj.count(k) != 0;
    }
    const Json& operator[](const std::string& k) const
    {
        static const Json kNull;
        auto it = obj.find(k);
        return it == obj.end() ? kNull : it->second;
    }
    const Json& operator[](std::size_t i) const
    {
        static const Json kNull;
        return i < arr.size() ? arr[i] : kNull;
    }
    std::size_t size() const { return kind == Kind::Array ? arr.size() : 0; }
    int as_int(int def = 0) const
    {
        return kind == Kind::Number ? static_cast<int>(num) : def;
    }
    double as_double(double def = 0.0) const { return kind == Kind::Number ? num : def; }
    std::string as_string(const std::string& def = "") const
    {
        return kind == Kind::String ? str : def;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text) {}
    Json parse()
    {
        skip_ws();
        return parse_value();
    }

private:
    const std::string& s_;
    std::size_t i_ = 0;

    void skip_ws()
    {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    char peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }

    Json parse_value()
    {
        skip_ws();
        switch (peek()) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': { Json j; j.kind = Json::Kind::String; j.str = parse_string(); return j; }
            case 't': i_ += 4; { Json j; j.kind = Json::Kind::Bool; j.b = true; return j; }
            case 'f': i_ += 5; { Json j; j.kind = Json::Kind::Bool; j.b = false; return j; }
            case 'n': i_ += 4; return Json{};
            default:  return parse_number();
        }
    }
    std::string parse_string()
    {
        ++i_;  // opening quote
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c == '\\' && i_ < s_.size()) {
                char e = s_[i_++];
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'u': i_ += 4; out += '?'; break;  // not needed for ASCII asset names
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        ++i_;  // closing quote
        return out;
    }
    Json parse_number()
    {
        std::size_t start = i_;
        if (peek() == '-') ++i_;
        while (i_ < s_.size() &&
              (std::isdigit(static_cast<unsigned char>(s_[i_])) || s_[i_] == '.' ||
               s_[i_] == 'e' || s_[i_] == 'E' || s_[i_] == '+' || s_[i_] == '-')) {
            ++i_;
        }
        Json j;
        j.kind = Json::Kind::Number;
        j.num  = std::stod(s_.substr(start, i_ - start));
        return j;
    }
    Json parse_array()
    {
        Json j;
        j.kind = Json::Kind::Array;
        ++i_;
        skip_ws();
        if (peek() == ']') { ++i_; return j; }
        while (true) {
            j.arr.push_back(parse_value());
            skip_ws();
            if (peek() == ',') { ++i_; continue; }
            break;
        }
        skip_ws();
        if (peek() == ']') ++i_;
        return j;
    }
    Json parse_object()
    {
        Json j;
        j.kind = Json::Kind::Object;
        ++i_;
        skip_ws();
        if (peek() == '}') { ++i_; return j; }
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (peek() == ':') ++i_;
            Json val = parse_value();
            j.obj.emplace(std::move(key), std::move(val));
            skip_ws();
            if (peek() == ',') { ++i_; skip_ws(); continue; }
            break;
        }
        skip_ws();
        if (peek() == '}') ++i_;
        return j;
    }
};

// ---------------------------------------------------------------------------
// Minimal double-precision linear algebra, column-major like glTF's `matrix`.
// ---------------------------------------------------------------------------
struct Vec3 { double x = 0, y = 0, z = 0; };
Vec3 operator*(Vec3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

struct Mat4 {
    double m[16]{};
    static Mat4 identity()
    {
        Mat4 r;
        for (int i = 0; i < 4; ++i) r.m[i * 4 + i] = 1.0;
        return r;
    }
};

Mat4 mul(const Mat4& a, const Mat4& b)
{
    Mat4 r{};
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            double sum = 0.0;
            for (int k = 0; k < 4; ++k) sum += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = sum;
        }
    }
    return r;
}

Vec3 transform_point(const Mat4& m, Vec3 p)
{
    return {m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z + m.m[12],
            m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z + m.m[13],
            m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z + m.m[14]};
}

// Linear part only - correct for normals under the orthogonal (rotation,
// optionally improper) transforms this asset's node hierarchy actually uses.
Vec3 transform_dir(const Mat4& m, Vec3 p)
{
    return {m.m[0] * p.x + m.m[4] * p.y + m.m[8] * p.z,
            m.m[1] * p.x + m.m[5] * p.y + m.m[9] * p.z,
            m.m[2] * p.x + m.m[6] * p.y + m.m[10] * p.z};
}

Vec3 normalize3(Vec3 v)
{
    double len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return len > 1e-12 ? v * (1.0 / len) : Vec3{0, 1, 0};
}

Mat4 mat_from_trs(Vec3 t, double qx, double qy, double qz, double qw, Vec3 s)
{
    const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const double wx = qw * qx, wy = qw * qy, wz = qw * qz;
    Mat4 r{};
    r.m[0] = (1 - 2 * (yy + zz)) * s.x; r.m[1] = 2 * (xy + wz) * s.x;       r.m[2] = 2 * (xz - wy) * s.x;       r.m[3] = 0;
    r.m[4] = 2 * (xy - wz) * s.y;       r.m[5] = (1 - 2 * (xx + zz)) * s.y; r.m[6] = 2 * (yz + wx) * s.y;       r.m[7] = 0;
    r.m[8] = 2 * (xz + wy) * s.z;       r.m[9] = 2 * (yz - wx) * s.z;       r.m[10] = (1 - 2 * (xx + yy)) * s.z; r.m[11] = 0;
    r.m[12] = t.x; r.m[13] = t.y; r.m[14] = t.z; r.m[15] = 1;
    return r;
}

struct BBox {
    Vec3 lo{1e30, 1e30, 1e30}, hi{-1e30, -1e30, -1e30};
    bool any = false;
    void add(Vec3 p)
    {
        any = true;
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
};

// ---------------------------------------------------------------------------
// GLB container + accessor reading
// ---------------------------------------------------------------------------
bool read_file(const std::string& path, std::vector<std::uint8_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0);
    out.resize(static_cast<std::size_t>(n));
    f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}

std::uint32_t read_u32(const std::uint8_t* p)
{
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

struct Accessor {
    int bufferView = -1, byteOffset = 0, componentType = 0, count = 0;
    std::string type;
};

Accessor get_accessor(const Json& root, int idx)
{
    const Json& a = root["accessors"][static_cast<std::size_t>(idx)];
    Accessor out;
    out.bufferView    = a["bufferView"].as_int(-1);
    out.byteOffset    = a["byteOffset"].as_int(0);
    out.componentType = a["componentType"].as_int(0);
    out.count         = a["count"].as_int(0);
    out.type          = a["type"].as_string();
    return out;
}

int type_components(const std::string& t)
{
    if (t == "SCALAR") return 1;
    if (t == "VEC2") return 2;
    if (t == "VEC3") return 3;
    if (t == "VEC4") return 4;
    return 0;
}

int component_size(int ct)
{
    switch (ct) {
        case 5120: case 5121: return 1;  // (UNSIGNED_)BYTE
        case 5122: case 5123: return 2;  // (UNSIGNED_)SHORT
        case 5125: case 5126: return 4;  // UNSIGNED_INT / FLOAT
        default: return 0;
    }
}

// Reads a float accessor (POSITION/NORMAL are always componentType FLOAT in
// this asset). `stride` defaults to tightly packed when the bufferView does
// not specify one explicitly.
std::vector<double> read_floats(const Json& root, const std::vector<std::uint8_t>& bin,
                                int accessor_idx)
{
    const Accessor a = get_accessor(root, accessor_idx);
    const Json& bv   = root["bufferViews"][static_cast<std::size_t>(a.bufferView)];
    const int bv_offset = bv["byteOffset"].as_int(0);
    const int ncomp      = type_components(a.type);
    const int csize       = component_size(a.componentType);
    int stride            = bv["byteStride"].as_int(0);
    if (stride == 0) stride = ncomp * csize;

    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(a.count) * ncomp);
    for (int i = 0; i < a.count; ++i) {
        const std::size_t base =
            static_cast<std::size_t>(bv_offset + a.byteOffset) + static_cast<std::size_t>(i) * stride;
        for (int c = 0; c < ncomp; ++c) {
            const std::size_t p = base + static_cast<std::size_t>(c) * csize;
            float v = 0.0f;
            std::memcpy(&v, &bin[p], 4);
            out.push_back(v);
        }
    }
    return out;
}

std::vector<std::uint32_t> read_indices(const Json& root, const std::vector<std::uint8_t>& bin,
                                        int accessor_idx)
{
    const Accessor a = get_accessor(root, accessor_idx);
    const Json& bv   = root["bufferViews"][static_cast<std::size_t>(a.bufferView)];
    const int bv_offset = bv["byteOffset"].as_int(0);
    const int csize       = component_size(a.componentType);
    int stride            = bv["byteStride"].as_int(0);
    if (stride == 0) stride = csize;

    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(a.count));
    for (int i = 0; i < a.count; ++i) {
        const std::size_t p =
            static_cast<std::size_t>(bv_offset + a.byteOffset) + static_cast<std::size_t>(i) * stride;
        std::uint32_t v = 0;
        if (a.componentType == 5125) {
            std::memcpy(&v, &bin[p], 4);
        } else if (a.componentType == 5123) {
            std::uint16_t v16 = 0;
            std::memcpy(&v16, &bin[p], 2);
            v = v16;
        } else {
            v = bin[p];  // UNSIGNED_BYTE
        }
        out.push_back(v);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Scene walk
// ---------------------------------------------------------------------------
struct BakedVertex { double px, py, pz, nx, ny, nz; };
struct Group {
    std::vector<BakedVertex>    verts;
    std::vector<std::uint32_t> idx;
};

Mat4 node_local_matrix(const Json& node)
{
    if (node.has("matrix")) {
        Mat4 m{};
        for (int i = 0; i < 16; ++i) m.m[i] = node["matrix"][static_cast<std::size_t>(i)].as_double(0.0);
        return m;
    }
    Vec3 t{0, 0, 0}, s{1, 1, 1};
    double qx = 0, qy = 0, qz = 0, qw = 1;
    if (node.has("translation")) {
        t = {node["translation"][0].as_double(0), node["translation"][1].as_double(0),
             node["translation"][2].as_double(0)};
    }
    if (node.has("rotation")) {
        qx = node["rotation"][0].as_double(0);
        qy = node["rotation"][1].as_double(0);
        qz = node["rotation"][2].as_double(0);
        qw = node["rotation"][3].as_double(1);
    }
    if (node.has("scale")) {
        s = {node["scale"][0].as_double(1), node["scale"][1].as_double(1), node["scale"][2].as_double(1)};
    }
    return mat_from_trs(t, qx, qy, qz, qw, s);
}

// Mesh-name prefixes that make up the main hull BODY - the keel, planking,
// frames, stanchions and stringer that run the length of the boat - as
// opposed to the mast, rigging and sail (which sit far above the hull) or the
// stem/sternposts, rudder and tiller (which are appendages that curl or jut
// out well beyond the hull body's own top and stern). All of those would
// otherwise skew the bounding box: a Viking ship's stem posts curl up far
// higher than its actual gunwale, so including them pulled the computed
// "hull top" - and with it the waterline this feeds into - up above nearly
// the entire real hull, leaving it rendering fully submerged. Used only to
// size and vertically anchor the physics hull; the RENDER split below is
// simpler (everything non-sail is one "hull" group, sail is the other) since
// the mast, rigging and stems use the same wood material as the hull anyway
// and still render correctly regardless of this list.
constexpr const char* kHullShellPrefixes[] = {
    "Keel", "Planking", "Frames", "Stanchions", "Stringer",
};

bool is_hull_shell(const std::string& mesh_name)
{
    for (const char* prefix : kHullShellPrefixes) {
        if (mesh_name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

void visit(const Json& root, const std::vector<std::uint8_t>& bin, int node_idx, Mat4 parent,
          Group& hull, Group& sail, std::vector<Vec3>& hull_shell_positions)
{
    const Json& node = root["nodes"][static_cast<std::size_t>(node_idx)];
    const Mat4 world = mul(parent, node_local_matrix(node));

    if (node.has("mesh")) {
        const Json& mesh = root["meshes"][static_cast<std::size_t>(node["mesh"].as_int(-1))];
        const std::string mesh_name = mesh["name"].as_string();
        const bool shell = is_hull_shell(mesh_name);

        for (std::size_t p = 0; p < mesh["primitives"].size(); ++p) {
            const Json& prim  = mesh["primitives"][p];
            const Json& attrs = prim["attributes"];
            if (!attrs.has("POSITION") || !attrs.has("NORMAL") || !prim.has("indices")) continue;

            const std::vector<double> pos = read_floats(root, bin, attrs["POSITION"].as_int());
            const std::vector<double> nrm = read_floats(root, bin, attrs["NORMAL"].as_int());
            const std::vector<std::uint32_t> idx = read_indices(root, bin, prim["indices"].as_int());
            // Material 1 is "Sail_Full" in this asset; everything else (hull,
            // mast, rigging, rudder) is material 0, "Ship_Oak_Aged".
            const int material = prim.has("material") ? prim["material"].as_int(-1) : -1;
            Group& target = (material == 1) ? sail : hull;

            const std::uint32_t base = static_cast<std::uint32_t>(target.verts.size());
            const std::size_t count = pos.size() / 3;
            for (std::size_t i = 0; i < count; ++i) {
                const Vec3 lp{pos[i * 3 + 0], pos[i * 3 + 1], pos[i * 3 + 2]};
                const Vec3 ln{nrm[i * 3 + 0], nrm[i * 3 + 1], nrm[i * 3 + 2]};
                const Vec3 wp = transform_point(world, lp);
                const Vec3 wn = normalize3(transform_dir(world, ln));
                target.verts.push_back({wp.x, wp.y, wp.z, wn.x, wn.y, wn.z});
                if (shell) hull_shell_positions.push_back(wp);
            }
            for (std::uint32_t id : idx) target.idx.push_back(base + id);
        }
    }

    for (std::size_t i = 0; i < node["children"].size(); ++i) {
        visit(root, bin, node["children"][i].as_int(), world, hull, sail, hull_shell_positions);
    }
}

void write_array_header(std::ofstream& out, const char* type, const char* name,
                        std::size_t count, const std::function<void(std::size_t)>& write_one)
{
    out << "inline constexpr " << type << ' ' << name << "[] = {\n";
    for (std::size_t i = 0; i < count; ++i) {
        write_one(i);
        if ((i + 1) % 12 == 0) out << '\n';
    }
    out << "\n};\n";
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: gltf_bake <model.glb> <output.h>\n");
        return 1;
    }

    std::vector<std::uint8_t> file;
    if (!read_file(argv[1], file)) {
        std::fprintf(stderr, "gltf_bake: cannot read %s\n", argv[1]);
        return 1;
    }
    if (file.size() < 12 || read_u32(&file[0]) != 0x46546C67u) {
        std::fprintf(stderr, "gltf_bake: %s is not a GLB file\n", argv[1]);
        return 1;
    }

    std::string json_text;
    std::vector<std::uint8_t> bin;
    std::size_t offset = 12;
    while (offset + 8 <= file.size()) {
        const std::uint32_t chunk_len  = read_u32(&file[offset]);
        const std::uint32_t chunk_type = read_u32(&file[offset + 4]);
        offset += 8;
        if (offset + chunk_len > file.size()) break;
        if (chunk_type == 0x4E4F534Au) {  // "JSON"
            json_text.assign(reinterpret_cast<const char*>(&file[offset]), chunk_len);
        } else if (chunk_type == 0x004E4942u) {  // "BIN\0"
            bin.assign(file.begin() + static_cast<std::ptrdiff_t>(offset),
                      file.begin() + static_cast<std::ptrdiff_t>(offset + chunk_len));
        }
        offset += chunk_len;
    }
    if (json_text.empty() || bin.empty()) {
        std::fprintf(stderr, "gltf_bake: missing JSON or BIN chunk in %s\n", argv[1]);
        return 1;
    }

    const Json root = JsonParser(json_text).parse();
    const int scene_idx = root.has("scene") ? root["scene"].as_int(0) : 0;
    const Json& scene = root["scenes"][static_cast<std::size_t>(scene_idx)];

    Group hull, sail;
    std::vector<Vec3> hull_shell_positions;
    const Mat4 identity = Mat4::identity();
    for (std::size_t i = 0; i < scene["nodes"].size(); ++i) {
        visit(root, bin, scene["nodes"][i].as_int(), identity, hull, sail, hull_shell_positions);
    }
    if (hull_shell_positions.empty()) {
        std::fprintf(stderr, "gltf_bake: no hull-shell meshes matched by name in %s\n", argv[1]);
        return 1;
    }

    // Raw (pre-scale) hull-shell extents decide which horizontal axis is the
    // boat's length: whichever of X/Z is larger. The physics probes in
    // floaters.hpp's Boat assume length runs along local X and beam along Z,
    // so if this model's long axis came out of the exporter as Z, a 90-degree
    // rotation about Y is folded into every vertex below to align it - this
    // works for ANY glTF asset dropped in here, not just this one.
    BBox raw;
    for (const Vec3& p : hull_shell_positions) raw.add(p);
    const double raw_size_x = raw.hi.x - raw.lo.x;
    const double raw_size_z = raw.hi.z - raw.lo.z;
    const bool swap_xz = raw_size_z > raw_size_x;
    const double length_axis = swap_xz ? raw_size_z : raw_size_x;

    // Target a small open Viking boat: bigger and more substantial on screen
    // than the old 6.4 m primitive hull (and than this model's first pass at
    // 7.5 m, which still read as small against the ocean's 800 m swells),
    // without inflating it into something implausible.
    constexpr double kTargetLength = 9.0;
    const double scale = kTargetLength / length_axis;

    auto remap = [&](Vec3 v) {
        if (swap_xz) v = {-v.z, v.y, v.x};
        return v * scale;
    };

    // Re-derive the hull-shell box AFTER the axis fix and scale, to get an
    // accurate half-extent and a centre to recentre every vertex on. Doing
    // this from the actual transformed shell points (rather than remapping
    // the raw box's two corners) is what stays correct regardless of which
    // axis got swapped or negated above.
    BBox shell;
    for (const Vec3& p : hull_shell_positions) shell.add(remap(p));
    const Vec3 half{(shell.hi.x - shell.lo.x) * 0.5, (shell.hi.y - shell.lo.y) * 0.5,
                    (shell.hi.z - shell.lo.z) * 0.5};
    const Vec3 center{(shell.hi.x + shell.lo.x) * 0.5, (shell.hi.y + shell.lo.y) * 0.5,
                      (shell.hi.z + shell.lo.z) * 0.5};

    // Where the waterline sits on the hull at rest, measured AMIDSHIPS rather
    // than against the whole bounding box.
    //
    // The box is the wrong ruler for this and the reason is the same one that
    // made kHullShellPrefixes necessary: a double-ender's gunwale sweeps up
    // toward both ends, so the box's height is the hull's depth AT THE ENDS,
    // which is the deepest part of it. Sizing the waterline against that gives
    // a boat far less freeboard amidships than intended - this hull came out
    // with about 0.45 m of it, and in a 12 m/s sea that is a hull the waves
    // close over rather than one that rides them.
    //
    // So take the section that actually decides how wet the boat gets: the
    // middle fifth of its length. `kDraftFraction` of THAT section's depth is
    // under water at rest, and the rest is freeboard.
    constexpr double kMidshipSpan   = 0.20;  // fraction of length counted as amidships
    constexpr double kDraftFraction = 0.55;

    BBox mid;
    const double mid_limit = kMidshipSpan * half.x;
    for (const Vec3& p : hull_shell_positions) {
        const Vec3 q = remap(p);
        if (std::fabs(q.x - center.x) <= mid_limit) mid.add(q);
    }
    if (!(mid.hi.y > mid.lo.y)) {
        std::fprintf(stderr, "gltf_bake: no hull-shell geometry amidships\n");
        return 1;
    }

    // Everything below is in hull-centre-relative coordinates, because that is
    // what the vertices are recentred to further down.
    const double mid_keel  = mid.lo.y - center.y;          // negative
    const double mid_depth = mid.hi.y - mid.lo.y;
    const double waterline = mid_keel + kDraftFraction * mid_depth;

    // Buoyancy is the restoring acceleration per metre of submersion, and the
    // viewer measures that submersion from the KEEL. So at rest the mean
    // submersion is (waterline - keel), and holding the hull there against
    // gravity needs exactly g over that depth - which is also g/draft, the
    // correct stiffness for a wall-sided hull floating at that draft, rather
    // than a number picked to land the boat at a height.
    //
    // That it falls out right matters: measuring submersion from the hull
    // CENTRE instead, as this used to, made the equilibrium depth 0.24 m on a
    // hull that actually draws 0.54 m, so the heave spring came out 2.2x too
    // stiff and the boat bobbed with a 0.99 s period - against 1.47 s here,
    // and no 9 m boat heaves in a second.
    const double draft    = waterline - (-half.y);
    const double buoyancy = 9.81 / draft;

    // The hull's SECTION TABLE: half-beam on a (station, level) grid covering
    // the whole hull, bow to stern and keel to sheer.
    //
    // This is what the ocean shader carves itself away with, so that the sea
    // stops at the hull instead of running straight through it and filling the
    // boat (ADR-028). An open boat is the case where that shows: a decked hull
    // hides the water inside it behind its own deck, and this one has nothing
    // to hide it with.
    //
    // WHY A TABLE and not a waterline profile plus an analytic taper. That was
    // tried, twice. A single half-beam per station, scaled toward the keel by
    // sqrt(depth), carves a shape far fatter than a sharp-bilged faering below
    // the waterline - and the error is invisible while the boat sits level,
    // because the sea is then AT the waterline where the profile is exact. It
    // shows the moment the hull lifts on a crest: the sea under it is then well
    // below the waterline, the over-wide carve reaches past the planking, and a
    // wedge of sky opens up alongside the hull. Guessing the section shape is
    // guessing at the very thing the mesh already knows, so this reads it.
    constexpr int kStations = 16;   // along the length
    constexpr int kLevels   = 8;    // keel to sheer
    // SIGNED bounds, not a symmetric half-beam.
    //
    // A half-beam assumes the hull is symmetric about the bounding box's
    // centreline, and this one is not: a faering carries its steering oar on
    // one quarter, which pushes the box's centre off the hull's own centreline
    // and makes one side of every section wider than the other. Carving with
    // the wider of the two over-carves the narrow side by the difference, and
    // that showed as a wedge of sky opening along one side of the hull and not
    // the other - the asymmetry of the artifact being the clue to the
    // asymmetry of the cause.
    double zlo[kStations][kLevels], zhi[kStations][kLevels];
    bool   seen[kStations][kLevels] = {};
    for (int st = 0; st < kStations; ++st)
        for (int lv = 0; lv < kLevels; ++lv) { zlo[st][lv] = 0.0; zhi[st][lv] = 0.0; }

    for (const Vec3& raw : hull_shell_positions) {
        const Vec3 q = remap(raw);
        const double lx = q.x - center.x, ly = q.y - center.y, lz = q.z - center.z;
        int st = static_cast<int>((lx + half.x) / (2.0 * half.x) * (kStations - 1) + 0.5);
        int lv = static_cast<int>((ly + half.y) / (2.0 * half.y) * (kLevels - 1) + 0.5);
        st = st < 0 ? 0 : (st >= kStations ? kStations - 1 : st);
        lv = lv < 0 ? 0 : (lv >= kLevels ? kLevels - 1 : lv);
        if (!seen[st][lv]) { zlo[st][lv] = zhi[st][lv] = lz; seen[st][lv] = true; }
        else { if (lz < zlo[st][lv]) zlo[st][lv] = lz;
               if (lz > zhi[st][lv]) zhi[st][lv] = lz; }
    }
    // A cell can be empty simply because no vertex landed in it, and an empty
    // cell would carve nothing where the hull is solid. Widen upward first: a
    // hull does not narrow as it rises from the keel, so the level below is a
    // sound bound on both sides. Then fill along the length from both ends,
    // for the stations the stem and stern posts leave bare.
    for (int st = 0; st < kStations; ++st) {
        for (int lv = 1; lv < kLevels; ++lv) {
            if (!seen[st][lv] && seen[st][lv - 1]) {
                zlo[st][lv] = zlo[st][lv - 1]; zhi[st][lv] = zhi[st][lv - 1];
                seen[st][lv] = true;
            } else if (seen[st][lv] && seen[st][lv - 1]) {
                zlo[st][lv] = std::min(zlo[st][lv], zlo[st][lv - 1]);
                zhi[st][lv] = std::max(zhi[st][lv], zhi[st][lv - 1]);
            }
        }
    }
    for (int lv = 0; lv < kLevels; ++lv) {
        for (int st = 1; st < kStations; ++st)
            if (!seen[st][lv] && seen[st - 1][lv]) {
                zlo[st][lv] = zlo[st - 1][lv]; zhi[st][lv] = zhi[st - 1][lv];
                seen[st][lv] = true;
            }
        for (int st = kStations - 2; st >= 0; --st)
            if (!seen[st][lv] && seen[st + 1][lv]) {
                zlo[st][lv] = zlo[st + 1][lv]; zhi[st][lv] = zhi[st + 1][lv];
                seen[st][lv] = true;
            }
    }

    // The end stations are the stem and stern posts, where the hull is a knife
    // edge - but they are also the stations most likely to have been empty and
    // filled from a neighbour, which hands a knife edge its neighbour's beam.
    // A boat comes to a point at both ends, so say so.
    for (int lv = 0; lv < kLevels; ++lv) {
        zlo[0][lv] = zhi[0][lv] = 0.0;
        zlo[kStations - 1][lv] = zhi[kStations - 1][lv] = 0.0;
    }

    // Report the asymmetry, because it is the whole reason these are signed.
    double worst_asym = 0.0;
    for (int st = 0; st < kStations; ++st)
        for (int lv = 0; lv < kLevels; ++lv)
            worst_asym = std::max(worst_asym, std::fabs(zhi[st][lv] + zlo[st][lv]));

    auto finalize = [&](const BakedVertex& v) {
        Vec3 p = remap({v.px, v.py, v.pz}) - center;
        Vec3 n = swap_xz ? Vec3{-v.nz, v.ny, v.nx} : Vec3{v.nx, v.ny, v.nz};
        return BakedVertex{p.x, p.y, p.z, n.x, n.y, n.z};
    };
    for (Group* g : {&hull, &sail}) {
        for (BakedVertex& v : g->verts) v = finalize(v);
    }

    std::ofstream out(argv[2], std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "gltf_bake: cannot open %s for writing\n", argv[2]);
        return 1;
    }
    // Fixed rather than default/scientific notation: every value here is a
    // small position or a unit normal component, and fixed guarantees a
    // decimal point in the output. Without one, an exact "1" or "-1" (a very
    // common normal component) would print as the literal `1f`, which MSVC
    // and GCC both reject - a float literal needs a '.' or exponent before
    // the suffix.
    out << std::fixed << std::setprecision(8);
    out << "// Generated by gltf_bake from " << argv[1] << ". Do not edit by hand.\n";
    out << "#pragma once\n\n#include <cstdint>\n\nnamespace viewer::boat_mesh {\n\n";
    out << "inline constexpr float kHalfExtent[3] = {" << half.x << "f, " << half.y << "f, "
       << half.z << "f};\n";
    out << "inline constexpr float kBuoyancy = " << buoyancy << "f;\n\n";
    out << "// Hull-local Y of the waterline at rest, and the hull's SIGNED z bounds on\n"
           "// a " << kStations << " x " << kLevels << " (station, level) grid spanning the whole hull:\n"
           "// station 0 is one end, level 0 the keel, level " << (kLevels - 1) << " the sheer. Signed\n"
           "// because the hull is not symmetric about its box centre. Pairs, lo then hi.\n"
           "// The ocean shader carves itself away inside this. See ADR-028.\n";
    out << "inline constexpr int kSectionStations = " << kStations << ";\n";
    out << "inline constexpr int kSectionLevels = " << kLevels << ";\n";
    out << "inline constexpr float kWaterlineY = " << waterline << "f;\n";
    out << "inline constexpr float kHullSection[" << (kStations * kLevels * 2) << "] = {";
    for (int st = 0; st < kStations; ++st) {
        for (int lv = 0; lv < kLevels; ++lv) {
            const int i = (st * kLevels + lv) * 2;
            out << (i ? "," : "") << "\n    " << zlo[st][lv] << "f, " << zhi[st][lv] << "f";
        }
    }
    out << "};\n\n";

    for (const auto& [name, g] : std::vector<std::pair<const char*, const Group*>>{
             {"Hull", &hull}, {"Sail", &sail}}) {
        write_array_header(out, "float", (std::string("k") + name + "Verts").c_str(),
                           g->verts.size() * 6, [&](std::size_t flat_i) {
                               const BakedVertex& v = g->verts[flat_i / 6];
                               const double comp[6] = {v.px, v.py, v.pz, v.nx, v.ny, v.nz};
                               out << comp[flat_i % 6] << "f,";
                           });
        write_array_header(out, "std::uint32_t", (std::string("k") + name + "Indices").c_str(),
                           g->idx.size(), [&](std::size_t i) { out << g->idx[i] << ','; });
    }

    out << "\n}  // namespace viewer::boat_mesh\n";

    std::fprintf(stderr,
                 "gltf_bake: hull %zu verts / %zu indices, sail %zu verts / %zu indices, "
                 "half-extent (%.3f, %.3f, %.3f) m\n"
                 "gltf_bake: hull section asymmetry about the box centreline %.3f m\n"
                 "gltf_bake: amidships depth %.3f m, waterline %+.3f m from hull centre, "
                 "draft %.3f m, freeboard %.3f m, buoyancy %.2f (heave period %.2f s)\n",
                 hull.verts.size(), hull.idx.size(), sail.verts.size(), sail.idx.size(), half.x,
                 half.y, half.z, worst_asym, mid_depth, waterline, draft,
                 (mid_keel + mid_depth) - waterline, buoyancy,
                 6.283185307 / std::sqrt(buoyancy));
    return 0;
}
