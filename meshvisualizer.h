#ifndef MESHVISUALIZER_H
#define MESHVISUALIZER_H

#include <QtOpenGL/QGLWidget>

#include "common.h"
#include "basicmesh.h"

class MeshVisualizer : public QGLWidget
{
public:
  MeshVisualizer(const string& title, const BasicMesh& mesh);

signals:

public slots:
  void initializeGL() override;
  void paintGL() override;
  void resizeGL(int w, int h) override;

private:
  BasicMesh mesh;
};

#endif // MESHVISUALIZER_H
