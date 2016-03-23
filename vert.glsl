//attribute highp vec4 vertex;
uniform highp mat4 modelview_matrix;
uniform highp mat4 normal_matrix;
uniform highp mat4 matrix;
varying vec3 vN;
varying vec3 v;

void main(void){
  gl_Position = matrix * gl_Vertex;
  v = modelview_matrix * gl_Vertex;
  vN = normal_matrix * gl_Vertex;
}
