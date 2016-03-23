#include <GL/freeglut_std.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>
#include "meshvisualizer2.h"

MeshVisualizer2::MeshVisualizer2(const string &title, const BasicMesh &mesh)
  : QGLWidget(QGLFormat(QGL::SampleBuffers | QGL::AlphaChannel | QGL::DepthBuffer)),
    mesh(mesh), image_tex(-1),
    use_external_rotation_translation(false),
    rot_x(0.0), rot_y(0.0), face_alpha(0.5),
    draw_faces(true), draw_edges(false), draw_points(false)
{
  setWindowTitle(title.c_str());
}

void MeshVisualizer2::initializeGL() {
  glShadeModel(GL_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_2D);

  if(!image.isNull()) {
    CreateTexture();
  }

  {
    shader_program.reset(new QOpenGLShaderProgram(this));
    shader_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "vert.glsl");
    shader_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "frag.glsl");
    shader_program->link();
  }
}

void MeshVisualizer2::CreateTexture() {
  cout << "Creating opengl texture ..." << endl;
#if 1
  if( image_tex >= 0 )
    glDeleteTextures(1, &image_tex);

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &image_tex);
  glBindTexture(GL_TEXTURE_2D, image_tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_RGBA,
               GL_UNSIGNED_BYTE, image.bits());
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
#else
  image_tex = bindTexture(pixmap);
#endif
  cout << "texture id = " << image_tex << endl;
  cout << "done." << endl;
}

void MeshVisualizer2::paintGL() {
  glClearColor(1, 1, 1, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);

  if( mesh.NumFaces() > 0 ) {
    // Draw image
#if 1
    if( !image.isNull() ) {
      glMatrixMode(GL_PROJECTION);
      glPushMatrix();
      glLoadIdentity();
      glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0001, 10.0);
      glViewport(0, 0, width(), height());

      glColor4f(1, 1, 1, 0.0);
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, image_tex);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
      glBegin(GL_QUADS);
      {
        glTexCoord2f(0.0f, 0.0f);
        glVertex3f(-1.0f, -1.0f, -5.0f);
        glTexCoord2f(1.0f, 0.0f);
        glVertex3f(1.0f, -1.0f, -5.0f);
        glTexCoord2f(1.0f, 1.0f);
        glVertex3f(1.0f, 1.0f, -5.0f);
        glTexCoord2f(0.0f, 1.0f);
        glVertex3f(-1.0f, 1.0f, -5.0f);
      }
      glEnd();
      glPopMatrix();
      glDisable(GL_TEXTURE_2D);
    }
#endif

    shader_program->bind();
    int matrixLocation = shader_program->uniformLocation("matrix");
    int modelview_matrix_location = shader_program->uniformLocation("modelview_matrix");
    int normal_matrix_location = shader_program->uniformLocation("normal_matrix");
    QMatrix4x4 pmvMatrix;
    QMatrix4x4 mvMatrix;
    QMatrix4x4 normalMatrix;

    // Setup Camera's model-view-projection matrix
    glm::dmat4 rotation_matrix;
    if(use_external_rotation_translation) {
      cout << "Using external rotation and translation ..." << endl;
      double aspect_ratio = width() / (double) height();

      double far = 1000.0, near = 0.01;
      glm::dmat4 Mproj = glm::dmat4(-camera_params.focal_length / (0.5 * camera_params.image_size.x), 0, 0, 0,
                                    0, -camera_params.focal_length / (0.5 * camera_params.image_size.y), 0, 0,
                                    0, 0, -(far+near)/(far-near), -1,
                                    0, 0, -2.0*far*near/(far-near), 0);


      glMatrixMode(GL_PROJECTION);
      glLoadMatrixd(&Mproj[0][0]);

      glViewport(0, 0, width(), height());

      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0), translation_vector_in);

      glm::dmat4 Rmat_interaction = glm::eulerAngleXY(rot_x, rot_y);

      glm::dmat4 Mmat = Tmat * Rmat_interaction * rotation_matrix_in;
      glMatrixMode(GL_MODELVIEW);

      glm::dmat4 Vmat = glm::lookAt(glm::dvec3(0, 0, 0),
                                    glm::dvec3(0, 0, -1),
                                    glm::dvec3(0, 1, 0));

      Vmat = glm::dmat4(1.0);

      glm::dmat4 MV = Vmat * Mmat;

      glLoadMatrixd(&MV[0][0]);

      rotation_matrix = rotation_matrix_in;

      glClear(GL_DEPTH_BUFFER_BIT);
    } else {
      glMatrixMode(GL_PROJECTION);

      const double aspect_ratio =
        camera_params.image_size.x / camera_params.image_size.y;

      const double far = camera_params.far;
      // near is the focal length
      const double near = camera_params.focal_length;
      const double top = near * tan(0.5 * camera_params.fovy);
      const double right = top * aspect_ratio;
      /*
      glm::dmat4 Mproj = glm::dmat4(near/right, 0, 0, 0,                            // col 0
                                    0, near/top, 0, 0,                              // col 1
                                    0, 0, -(far+near)/(far-near), -1,               // col 2
                                    0, 0, -2.0 * far * near / (far - near), 0.0);   // col 3
      */
      const double focal_length = camera_params.focal_length;
      glm::dmat4 Mproj = glm::dmat4(1, 0, 0, 0,
                                    0, 1*aspect_ratio, 0, 0,
                                    0, 0, 1/focal_length, -1/focal_length,
                                    0, 0, 0, 0);
      glLoadMatrixd(&Mproj[0][0]);

      glViewport(0, 0, width(), height());

      glm::dmat4 Rmat = glm::eulerAngleYXZ(mesh_rotation[0],
                                           mesh_rotation[1],
                                           mesh_rotation[2]);

      glm::dmat4 Rmat_interaction = glm::eulerAngleXY(rot_x, rot_y);

      glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                       glm::dvec3(mesh_translation[0],
                                                  mesh_translation[1],
                                                  mesh_translation[2]));

      glm::dmat4 MV = Tmat * Rmat_interaction * Rmat;
      glMatrixMode(GL_MODELVIEW);
      glLoadMatrixd(&MV[0][0]);

      rotation_matrix = Rmat;

      glm::dmat4 MVP = Mproj * MV;
      pmvMatrix = QMatrix4x4(MVP[0][0], MVP[1][0], MVP[2][0], MVP[3][0],
                             MVP[0][1], MVP[1][1], MVP[2][1], MVP[3][1],
                             MVP[0][2], MVP[1][2], MVP[2][2], MVP[3][2],
                             MVP[0][3], MVP[1][3], MVP[2][3], MVP[3][3]);
      mvMatrix = QMatrix4x4(MV[0][0], MV[1][0], MV[2][0], MV[3][0],
                            MV[0][1], MV[1][1], MV[2][1], MV[3][1],
                            MV[0][2], MV[1][2], MV[2][2], MV[3][2],
                            MV[0][3], MV[1][3], MV[2][3], MV[3][3]);
      normalMatrix = mvMatrix.inverted().transposed();
    }

    shader_program->setUniformValue(matrixLocation, pmvMatrix);
    shader_program->setUniformValue(modelview_matrix_location, mvMatrix);
    shader_program->setUniformValue(normal_matrix_location, normalMatrix);

    glPushMatrix();

    EnableLighting();

    if( draw_faces ) {

      glEnable(GL_CULL_FACE);
      glCullFace(GL_BACK);

      /// Draw faces
      glColor4d(.75, .75, .75, face_alpha);
      GLfloat mat_diffuse[] = {0.5, 0.5, 0.5, static_cast<float>(face_alpha)};
      GLfloat mat_specular[] = {0.25, 0.25, 0.25, static_cast<float>(face_alpha)};
      GLfloat mat_shininess[] = {75.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);

      glBegin(GL_TRIANGLES);
      for (int i = 0; i < mesh.NumFaces(); ++i) {
        auto face_i = mesh.face(i);
        auto v0 = mesh.vertex(face_i[0]);
        auto v1 = mesh.vertex(face_i[1]);
        auto v2 = mesh.vertex(face_i[2]);
        auto n0 = mesh.vertex_normal(face_i[0]);
        auto n1 = mesh.vertex_normal(face_i[1]);
        auto n2 = mesh.vertex_normal(face_i[2]);

        auto set_diffuse_color_by_normal = [=](Vector3d n0) {
          glm::dvec4 n = glm::transpose(glm::inverse(rotation_matrix)) * glm::dvec4(n0[0], n0[1], n0[2], 1.0);
          if(n.z > 0) {
            GLfloat mat_diffuse[] = {static_cast<float>(n.x + 1.0) * 0.5f,
                                     static_cast<float>(n.y + 1.0) * 0.5f,
                                     static_cast<float>(n.z + 1.0) * 0.5f,
                                     static_cast<float>(face_alpha)};
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
          } else {
            GLfloat mat_diffuse[] = {0.05, 0.05, 0.05, static_cast<float>(face_alpha)};
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
          }
        };

        //set_diffuse_color_by_normal(n0);
        glNormal3dv(n0.data());glVertex3dv(v0.data());

        //set_diffuse_color_by_normal(n1);
        glNormal3dv(n1.data());glVertex3dv(v1.data());

        //set_diffuse_color_by_normal(n2);
        glNormal3dv(n2.data());glVertex3dv(v2.data());
      }
      glEnd();
      glDisable(GL_CULL_FACE);
    }

    if( draw_edges ) {
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

    // Draw landmarks
    if (draw_points) {
      cout << "landmarks:" << endl;
      glColor3f(.75, .25, .25);
      GLfloat mat_diffuse[] = {0.875, 0.375, 0.375, 1.0};
      GLfloat mat_specular[] = {0.875, 0.875, 0.875, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
      glPointSize(3.0);
      for (auto &landmark : landmarks) {
        auto v = mesh.vertex(landmark);
        const double delta_z = 1e-2;
        glPushMatrix();
        cout << landmark << ": " << v.transpose() << endl;
        glTranslated(v[0], v[1], v[2] + delta_z);
        glutSolidSphere(0.01, 32, 32);
        glPopMatrix();
      }
    }

    // Draw updated landmarks
    if( draw_points ) {
      cout << "updated landmarks:" << endl;
      glColor3f(.25, .25, .75);
      GLfloat mat_diffuse[] = {0.375, 0.375, 0.875, 1.0};
      GLfloat mat_specular[] = {0.875, 0.875, 0.875, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
      glPointSize(3.0);
      for (auto &landmark : updated_landmarks) {
        auto v = mesh.vertex(landmark);
        const double delta_z = 1e-2;
        glPushMatrix();
        cout << landmark << ": " << v.transpose() << endl;
        glTranslated(v[0], v[1], v[2] + delta_z);
        glutSolidSphere(0.01, 32, 32);
        glPopMatrix();
      }
    }

    glPopMatrix();

    shader_program->release();

    // Draw constraints
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, width(), 0, height(), 0.0001, 1000.0);
    glViewport(0, 0, width(), height());

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    if( draw_points ) {
      glColor3f(.25, .75, .25);
      GLfloat mat_diffuse[] = {0.375, 0.875, 0.375, 1.0};
      GLfloat mat_specular[] = {0.875, 0.875, 0.875, 1.0};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
      glPointSize(3.0);
      for (int i = 0; i < constraints.size(); ++i) {
        auto &constraint = constraints[i];
        double xi = constraint.data.x;
        double yi = constraint.data.y;

        // transform xi and yi if necessary
        double xratio = static_cast<double>(width()) / static_cast<double>(image.width());
        double yratio = static_cast<double>(height()) / static_cast<double>(image.height());

        xi *= xratio;
        yi *= yratio;

        glPushMatrix();
        cout << i << ": " << constraint.data.x << ", " << constraint.data.y <<
        endl;
        glTranslated(xi, yi, 2.0);
        glutSolidSphere(3, 32, 32);
        glPopMatrix();
      }
    }

    DisableLighting();
  } else {
    cout << "no mesh" << endl;
    glColor3f(1, 0, 0);
    glPushMatrix();
    glLineWidth(2.5);
    glColor3f(0.25, 0.75, 0.25);
    double square_size = 1.0;
    glBegin(GL_LINE_LOOP);
    glVertex3d(-square_size, -square_size, 0.0);
    glVertex3d( square_size, -square_size, 0.0);
    glVertex3d( square_size,  square_size, 0.0);
    glVertex3d(-square_size,  square_size, 0.0);
    glEnd();
    glPopMatrix();
  }
}

void MeshVisualizer2::resizeGL(int w, int h) {
  QGLWidget::resizeGL(w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45.0, static_cast<float>(w)/ static_cast<float>(h), 0.0001, 1000.00);
  glViewport(0, 0, w, h);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void MeshVisualizer2::EnableLighting()
{
  GLfloat light_position[] = {10.0, 4.0, 10.0, 1.0};
  GLfloat mat_specular[] = {0.5, 0.5, 0.5, 1.0};
  GLfloat mat_diffuse[] = {0.375, 0.375, 0.375, 1.0};
  GLfloat mat_shininess[] = {25.0};
  GLfloat light_ambient[] = {0.05, 0.05, 0.05, 1.0};
  GLfloat white_light[] = {1.0, 1.0, 1.0, 1.0};

  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);

  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT0, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);

  light_position[0] = -10.0;
  glLightfv(GL_LIGHT1, GL_POSITION, light_position);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
  glLightfv(GL_LIGHT1, GL_SPECULAR, white_light);
  glLightfv(GL_LIGHT1, GL_AMBIENT, light_ambient);

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
}

void MeshVisualizer2::DisableLighting()
{
  glDisable(GL_LIGHT0);
  glDisable(GL_LIGHT1);
  glDisable(GL_LIGHTING);
}

void MeshVisualizer2::BindConstraints(const vector<Constraint2D> &constraints_in) {
  constraints = constraints_in;
}

void MeshVisualizer2::BindImage(const QImage& img) {
  image = convertToGLFormat(img);
  CreateTexture();
}

void MeshVisualizer2::BindLandmarks(const vector<int> &landmarks_in) {
  landmarks = landmarks_in;
}

void MeshVisualizer2::SetMeshRotationTranslation(const Vector3d &R,
                                                const Vector3d &T) {
  mesh_rotation = R;
  mesh_translation = T;
}

void MeshVisualizer2::SetCameraParameters(const CameraParameters &cam_params) {
  camera_params = cam_params;
}

void MeshVisualizer2::keyPressEvent(QKeyEvent *event) {
  switch(event->key()) {
    case Qt::Key_F: {
      draw_faces = !draw_faces;
      repaint();
      event->accept();
      break;
    }
    case Qt::Key_Equal:
    case Qt::Key_Plus: {
      face_alpha += 0.05;
      face_alpha = min(face_alpha, 1.0);
      repaint();
      event->accept();
      break;
    }
    case Qt::Key_hyphen:
    case Qt::Key_Minus: {
      face_alpha -= 0.05;
      face_alpha = max(face_alpha, 0.05);
      repaint();
      event->accept();
      break;
    }
    case Qt::Key_Left:
    case Qt::Key_Right: {
      double delta = 0.05 * (event->key()==Qt::Key_Left?-1.0:1.0);
      rot_y += delta;
      repaint();
      event->accept();
      break;
    }
    case Qt::Key_Up:
    case Qt::Key_Down: {
      double delta = 0.05 * (event->key()==Qt::Key_Down?-1.0:1.0);
      rot_x += delta;
      repaint();
      event->accept();
      break;
    }
    case Qt::Key_P: {
      draw_points = !draw_points;
      repaint();
      event->accept();
      break;
    }
  }
}

void MeshVisualizer2::BindUpdatedLandmarks(
  const vector<int> &updated_landmarks_in) {
  updated_landmarks = updated_landmarks_in;
}
