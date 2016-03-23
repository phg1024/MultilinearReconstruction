#ifndef MESHVISUALIZER2_H
#define MESHVISUALIZER2_H

#include <QtOpenGL/QGLWidget>
#include <QOpenGLShaderProgram>
#include <QKeyEvent>

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "parameters.h"

class MeshVisualizer2 : public QGLWidget
{
public:
  MeshVisualizer2(const string& title, const BasicMesh& mesh);

signals:

public slots:
  void initializeGL() override;
  void paintGL() override;
  void resizeGL(int w, int h) override;

  void EnableLighting();
  void DisableLighting();

  void BindConstraints(const vector<Constraint2D>& constraints_in);
  void BindImage(const QImage& img);
  void BindLandmarks(const vector<int>& landmarks_in);
  void BindUpdatedLandmarks(const vector<int>& updated_landmarks_in);

  void SetMeshRotationTranslation(const Vector3d& R, const Vector3d& T);
  void SetCameraParameters(const CameraParameters& cam_params);

  void SetRotationMatrixTranslationVector(const glm::dmat4& R, const glm::dvec3& T) {
    use_external_rotation_translation = true;
    rotation_matrix_in = R;
    translation_vector_in = T;
  }

protected:
  void CreateTexture();

  void keyPressEvent(QKeyEvent* event);

private:
  BasicMesh mesh;
  vector<Constraint2D> constraints;
  QImage image;
  GLuint image_tex;

  shared_ptr<QOpenGLShaderProgram> shader_program;

  vector<int> landmarks;
  vector<int> updated_landmarks;

  Vector3d mesh_rotation, mesh_translation;
  CameraParameters camera_params;

  double rot_x, rot_y;

  bool use_external_rotation_translation;
  glm::dmat4 rotation_matrix_in;
  glm::dvec3 translation_vector_in;

  double face_alpha;
  bool draw_faces, draw_edges;
  bool draw_points;
};

#endif  // MESHVISUALIZER2_H
