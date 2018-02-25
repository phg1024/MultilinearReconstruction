#ifndef FACESHAPEFROMSHADING_OFFSCREENMESHVISUALIZER_H
#define FACESHAPEFROMSHADING_OFFSCREENMESHVISUALIZER_H

//#include "Geometry/geometryutils.hpp"
//#include "Utils/utility.hpp"

#include "basicmesh.h"
#include "parameters.h"

#include <QDir>
#include <QImage>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOffscreenSurface>

#include <boost/timer/timer.hpp>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

namespace ColorEncoding {
  inline void encode_index(int idx, unsigned char& r, unsigned char& g, unsigned char& b) {
    r = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    g = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    b = static_cast<unsigned char>(idx & 0xff);
  }

  inline int decode_index(unsigned char r, unsigned char g, unsigned char b, int& idx) {
    idx = b; idx <<= 8; idx |= g; idx <<= 8; idx |= r;
    return idx;
  }
}

class OffscreenMeshVisualizer {
public:
  enum MVPMode {
    OrthoNormal,
    OrthoNormalExtended,
    CamPerspective,
    BackgroundImage
  };
  enum RenderMode {
    Texture,
    BarycentricCoordinates,
    Normal,
    Mesh,
    MeshAndImage,
    TexturedMesh
  };
  OffscreenMeshVisualizer(int width, int height)
   : width(width), height(height), index_encoded(true), lighting_enabled(false) {
    // Load rendering settings
    {
      const string home_directory = QDir::homePath().toStdString();
      cout << "Home dir: " << home_directory << endl;

      ifstream fin(home_directory + "/Data/Settings/blendshape_vis_ao.json");
      fin >> rendering_settings;
    }
  }

  void LoadRenderingSettings(const string& filename) {
    ifstream fin(filename);
    fin >> rendering_settings;
  }

  void BindMesh(const BasicMesh& in_mesh) {
    mesh = in_mesh;
  }
  void BindTexture(const QImage& in_texture) {
    texture = in_texture;
  }
  void BindImage(const QImage& img) {
    image = img;
  }
  void SetMeshRotationTranslation(const Vector3d& R, const Vector3d& T) {
    mesh_rotation = R;
    mesh_translation = T;
  }
  void SetCameraParameters(const CameraParameters& cam_params) {
    camera_params = cam_params;
  }
  void SetFacesToRender(const vector<int>& indices) {
    faces_to_render = indices;
  }
  void SetNormals(const vector<float>& ns) {
    normals = ns;
  }
  void SetAmbientOcclusion(const vector<float>& ao_in) {
    ao = ao_in;
  }

  void SetRenderMode(RenderMode mode_in) {
    render_mode = mode_in;
  }
  void SetMVPMode(MVPMode mode_in) {
    mode = mode_in;
  }
  void SetIndexEncoded(bool val) {
    index_encoded = val;
  }
  void SetEnableLighting(bool val) {
    lighting_enabled = val;
  }

  QImage Render(bool multi_sampled=false) const;
  pair<QImage, vector<float>> RenderWithDepth(bool multi_sampled=false) const;

protected:
  void SetupViewing(const MVPMode&) const;
  void CreateTexture() const;
  void EnableLighting() const;
  void DisableLighting() const;

private:
  int width, height;
  MVPMode mode;
  RenderMode render_mode;

  Vector3d mesh_rotation, mesh_translation;
  CameraParameters camera_params;

  mutable vector<int> faces_to_render;
  mutable vector<float> normals;
  mutable vector<float> ao;

  bool index_encoded;
  bool lighting_enabled;
  BasicMesh mesh;
  QImage image;
  mutable GLuint image_tex;
  QImage texture;
  mutable glm::dmat4 Mview;

  json rendering_settings;
  mutable vector<GLuint> enabled_lights;
};


#endif //FACESHAPEFROMSHADING_OFFSCREENMESHVISUALIZER_H
