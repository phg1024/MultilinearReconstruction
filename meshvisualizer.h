#ifndef MESHVISUALIZER_H
#define MESHVISUALIZER_H

#include <QtOpenGL/QGLWidget>
#include <QKeyEvent>

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "parameters.h"

class MeshVisualizer : public QGLWidget
{
public:
  MeshVisualizer(const string& title, const BasicMesh& mesh);

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

protected:
  void CreateTexture();

  void keyPressEvent(QKeyEvent* event);

private:
  BasicMesh mesh;
  vector<Constraint2D> constraints;
  QImage image;
  GLuint image_tex;

  vector<int> landmarks;
  vector<int> updated_landmarks;

  Vector3d mesh_rotation, mesh_translation;
  CameraParameters camera_params;

  double rot_x, rot_y;

  double face_alpha;
  bool draw_faces, draw_edges;
  bool draw_points;
};

#endif // MESHVISUALIZER_H
