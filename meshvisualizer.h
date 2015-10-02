#ifndef MESHVISUALIZER_H
#define MESHVISUALIZER_H

#include <QtOpenGL/QGLWidget>

#include "common.h"
#include "basicmesh.h"
#include "constraints.h"

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

protected:
  void CreateTexture();

private:
  BasicMesh mesh;
  vector<Constraint2D> constraints;
  QImage image;
  GLuint image_tex;

  vector<int> landmarks;

  double face_alpha;
  bool draw_faces, draw_edges;
};

#endif // MESHVISUALIZER_H
