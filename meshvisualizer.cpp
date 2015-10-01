#include <GL/freeglut_std.h>
#include "meshvisualizer.h"

MeshVisualizer::MeshVisualizer(const string &title, const BasicMesh &mesh)
  : QGLWidget(QGLFormat(QGL::SampleBuffers | QGL::AlphaChannel)), mesh(mesh)
{
  setWindowTitle(title.c_str());
}

void MeshVisualizer::initializeGL() {
  glShadeModel(GL_SMOOTH);
}

void MeshVisualizer::paintGL() {
  glClearColor(1, 1, 1, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  if( mesh.NumFaces() > 0 ) {
    glPushMatrix();
    glTranslatef(0, 0, -5.0);

    {
      /// Draw faces
      glColor3f(.75, .75, .75);
      glBegin(GL_TRIANGLES);
      for (int i = 0; i < mesh.NumFaces(); ++i) {
        auto face_i = mesh.face(i);
        auto v0 = mesh.vertex(face_i[0]);
        auto v1 = mesh.vertex(face_i[1]);
        auto v2 = mesh.vertex(face_i[2]);
        auto n0 = mesh.normal(face_i[0]);
        auto n1 = mesh.normal(face_i[1]);
        auto n2 = mesh.normal(face_i[2]);
        glNormal3dv(n0.data());glVertex3dv(v0.data());
        glNormal3dv(n1.data());glVertex3dv(v1.data());
        glNormal3dv(n2.data());glVertex3dv(v2.data());
      }
      glEnd();
    }

    {
      /// Draw edges
      glColor3f(.25, .25, .25);
      glLineWidth(2.5);
      for (int i = 0; i < mesh.NumFaces(); ++i) {
        auto face_i = mesh.face(i);
        auto v0 = mesh.vertex(face_i[0]);
        auto v1 = mesh.vertex(face_i[1]);
        auto v2 = mesh.vertex(face_i[2]);
        glBegin(GL_LINE_LOOP);
        glVertex3dv(v0.data());
        glVertex3dv(v1.data());
        glVertex3dv(v2.data());
        glEnd();
      }
    }

    glPopMatrix();
  } else {
    glColor3f(1, 0, 0);
    glPushMatrix();
    glTranslatef(0, 0, -5.0);
    glutSolidTeapot(1.0);
    glPopMatrix();
  }
}

void MeshVisualizer::resizeGL(int w, int h) {
  QGLWidget::resizeGL(w, h);
  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60.0, static_cast<float>(w)/ static_cast<float>(h), 0.0001, 10.00);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}