/////////////////////////////////////////////
//
// Mesh Simplification Tutorial
//
// (C) by Sven Forstmann in 2014
//
// License : MIT
// http://opensource.org/licenses/MIT
//
//https://github.com/sp4cerat/Fast-Quadric-Mesh-S;
// 5/2016: Chris Rorden created minimal version for OSX/Linux/Windows compile

//#include <iostream>
//#include <stddef.h>
//#include <functional>
//#include <sys/stat.h>
//#include <stdbool.h>
#include <string.h>
//#include <ctype.h>
//#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <vector>
#include <string>
#include <math.h>
#include <float.h> //FLT_EPSILON, DBL_EPSILON

#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )
#define loopi(start_l,end_l) for ( int i=start_l;i<end_l;++i )
#define loopj(start_l,end_l) for ( int j=start_l;j<end_l;++j )
#define loopk(start_l,end_l) for ( int k=start_l;k<end_l;++k )

struct vector3
{
double x, y, z;
};

struct vec3f
{
    double x, y, z;

    inline vec3f( void ) {}

    //inline vec3f operator =( vector3 a )
  // { vec3f b ; b.x = a.x; b.y = a.y; b.z = a.z; return b;}

    inline vec3f( vector3 a )
   { x = a.x; y = a.y; z = a.z; }

    inline vec3f( const double X, const double Y, const double Z )
    { x = X; y = Y; z = Z; }

    inline vec3f operator + ( const vec3f& a ) const
    { return vec3f( x + a.x, y + a.y, z + a.z ); }

  inline vec3f operator += ( const vec3f& a ) const
    { return vec3f( x + a.x, y + a.y, z + a.z ); }

    inline vec3f operator * ( const double a ) const
    { return vec3f( x * a, y * a, z * a ); }

    inline vec3f operator * ( const vec3f a ) const
    { return vec3f( x * a.x, y * a.y, z * a.z ); }

    inline vec3f v3 () const
    { return vec3f( x , y, z ); }

    inline vec3f operator = ( const vector3 a )
    { x=a.x;y=a.y;z=a.z;return *this; }

    inline vec3f operator = ( const vec3f a )
    { x=a.x;y=a.y;z=a.z;return *this; }

    inline vec3f operator / ( const vec3f a ) const
    { return vec3f( x / a.x, y / a.y, z / a.z ); }

    inline vec3f operator - ( const vec3f& a ) const
    { return vec3f( x - a.x, y - a.y, z - a.z ); }

    inline vec3f operator / ( const double a ) const
    { return vec3f( x / a, y / a, z / a ); }

    inline double dot( const vec3f& a ) const
    { return a.x*x + a.y*y + a.z*z; }

    inline vec3f cross( const vec3f& a , const vec3f& b )
    {
    x = a.y * b.z - a.z * b.y;
    y = a.z * b.x - a.x * b.z;
    z = a.x * b.y - a.y * b.x;
    return *this;
  }

    inline double angle( const vec3f& v )
    {
    vec3f a = v , b = *this;
    double dot = v.x*x + v.y*y + v.z*z;
    double len = a.length() * b.length();
    if(len==0)len=0.00001f;
    double input = dot  / len;
    if (input<-1) input=-1;
    if (input>1) input=1;
    return (double) acos ( input );
  }

    inline double angle2( const vec3f& v , const vec3f& w )
    {
    vec3f a = v , b= *this;
    double dot = a.x*b.x + a.y*b.y + a.z*b.z;
    double len = a.length() * b.length();
    if(len==0)len=1;

    vec3f plane; plane.cross( b,w );

    if ( plane.x * a.x + plane.y * a.y + plane.z * a.z > 0 )
      return (double) -acos ( dot  / len );

    return (double) acos ( dot  / len );
  }

    inline vec3f rot_x( double a )
    {
    double yy = cos ( a ) * y + sin ( a ) * z;
    double zz = cos ( a ) * z - sin ( a ) * y;
    y = yy; z = zz;
    return *this;
  }
    inline vec3f rot_y( double a )
    {
    double xx = cos ( -a ) * x + sin ( -a ) * z;
    double zz = cos ( -a ) * z - sin ( -a ) * x;
    x = xx; z = zz;
    return *this;
  }
    inline void clamp( double min, double max )
    {
    if (x<min) x=min;
    if (y<min) y=min;
    if (z<min) z=min;
    if (x>max) x=max;
    if (y>max) y=max;
    if (z>max) z=max;
  }
    inline vec3f rot_z( double a )
    {
    double yy = cos ( a ) * y + sin ( a ) * x;
    double xx = cos ( a ) * x - sin ( a ) * y;
    y = yy; x = xx;
    return *this;
  }
    inline vec3f invert()
  {
    x=-x;y=-y;z=-z;return *this;
  }
    inline vec3f frac()
  {
    return vec3f(
      x-double(int(x)),
      y-double(int(y)),
      z-double(int(z))
      );
  }

    inline vec3f integer()
  {
    return vec3f(
      double(int(x)),
      double(int(y)),
      double(int(z))
      );
  }

    inline double length() const
    {
    return (double)sqrt(x*x + y*y + z*z);
  }

    inline vec3f normalize( double desired_length = 1 )
    {
    double square = sqrt(x*x + y*y + z*z);
    /*
    if (square <= 0.00001f )
    {
      x=1;y=0;z=0;
      return *this;
    }*/
    //double len = desired_length / square;
    x/=square;y/=square;z/=square;

    return *this;
  }
    static vec3f normalize( vec3f a );

  static void random_init();
  static double random_double();
  static vec3f random();

  static int random_number;

  double random_double_01(double a){
    double rnf=a*14.434252+a*364.2343+a*4213.45352+a*2341.43255+a*254341.43535+a*223454341.3523534245+23453.423412;
    int rni=((int)rnf)%100000;
    return double(rni)/(100000.0f-1.0f);
  }

  vec3f random01_fxyz(){
    x=(double)random_double_01(x);
    y=(double)random_double_01(y);
    z=(double)random_double_01(z);
    return *this;
  }

};

vec3f barycentric(const vec3f &p, const vec3f &a, const vec3f &b, const vec3f &c){
  vec3f v0 = b-a;
  vec3f v1 = c-a;
  vec3f v2 = p-a;
  double d00 = v0.dot(v0);
  double d01 = v0.dot(v1);
  double d11 = v1.dot(v1);
  double d20 = v2.dot(v0);
  double d21 = v2.dot(v1);
  double denom = d00*d11-d01*d01;
  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;
  return vec3f(u,v,w);
}

vec3f interpolate(const vec3f &p, const vec3f &a, const vec3f &b, const vec3f &c, const vec3f attrs[3])
{
  vec3f bary = barycentric(p,a,b,c);
  vec3f out = vec3f(0,0,0);
  out = out + attrs[0] * bary.x;
  out = out + attrs[1] * bary.y;
  out = out + attrs[2] * bary.z;
  return out;
}

double min(double v1, double v2) {
  return fmin(v1,v2);
}


class SymetricMatrix {

  public:

  // Constructor

  SymetricMatrix(double c=0) { loopi(0,10) m[i] = c;  }

  SymetricMatrix( double m11, double m12, double m13, double m14,
                  double m22, double m23, double m24,
                              double m33, double m34,
                                          double m44) {
       m[0] = m11;  m[1] = m12;  m[2] = m13;  m[3] = m14;
                    m[4] = m22;  m[5] = m23;  m[6] = m24;
                                 m[7] = m33;  m[8] = m34;
                                              m[9] = m44;
  }

  // Make plane

  SymetricMatrix(double a,double b,double c,double d)
  {
    m[0] = a*a;  m[1] = a*b;  m[2] = a*c;  m[3] = a*d;
                 m[4] = b*b;  m[5] = b*c;  m[6] = b*d;
                              m[7 ] =c*c; m[8 ] = c*d;
                                           m[9 ] = d*d;
  }

  double operator[](int c) const { return m[c]; }

  // Determinant

  double det( int a11, int a12, int a13,
        int a21, int a22, int a23,
        int a31, int a32, int a33)
  {
    double det =  m[a11]*m[a22]*m[a33] + m[a13]*m[a21]*m[a32] + m[a12]*m[a23]*m[a31]
          - m[a13]*m[a22]*m[a31] - m[a11]*m[a23]*m[a32]- m[a12]*m[a21]*m[a33];
    return det;
  }

  const SymetricMatrix operator+(const SymetricMatrix& n) const
  {
    return SymetricMatrix( m[0]+n[0],   m[1]+n[1],   m[2]+n[2],   m[3]+n[3],
                                m[4]+n[4],   m[5]+n[5],   m[6]+n[6],
                                             m[ 7]+n[ 7], m[ 8]+n[8 ],
                                                          m[ 9]+n[9 ]);
  }

  SymetricMatrix& operator+=(const SymetricMatrix& n)
  {
     m[0]+=n[0];   m[1]+=n[1];   m[2]+=n[2];   m[3]+=n[3];
     m[4]+=n[4];   m[5]+=n[5];   m[6]+=n[6];   m[7]+=n[7];
     m[8]+=n[8];   m[9]+=n[9];
    return *this;
  }

  // Scale matrix by a factor (e.g., border_weight)
  SymetricMatrix operator*(const double s) const {
    return SymetricMatrix(m[0] * s, m[1] * s, m[2] * s, m[3] * s,
                          m[4] * s, m[5] * s, m[6] * s,
                          m[7] * s, m[8] * s,
                          m[9] * s);
  }

  SymetricMatrix& operator*=(const double s) {
    loopi(0, 10) m[i] *= s;
    return *this;
  }

  double m[10];
};
///////////////////////////////////////////

namespace Simplify
{
  // ==========================================================================
  // [triflow] FORK ADDITIONS -- overview
  // --------------------------------------------------------------------------
  // This file is forked from upstream pyfqmr and customized for triflow.
  // Search this file for "[triflow]" to find every non-trivial modified block.
  //
  // Added fields on Vertex:   custom_q, area, tid
  // Added field on Triangle:  border[3]  (per-edge border flag)
  // Added globals:            custom_quadratics[], has_target_vertex_ids,
  //                           minimum_edge_length, border_weight,
  //                           binary_search_steps, override_quadratics
  // Added simplify_mesh args: custom_quadratics_param, target_vertex_ids,
  //                           min_edge_length
  // Modified behavior:        area-weighted quadrics, area-normalized error,
  //                           per-edge borders, non-manifold pinch check,
  //                           target-group split check, optimal-position
  //                           safety fallback, flipped() full-ring scan,
  //                           termination-on-no-progress.
  //
  // See README "What's different from upstream" for rationale.
  // ==========================================================================

  // Global Variables & Strctures
  enum Attributes {
    NONE,
    NORMAL = 2,
    TEXCOORD = 4,
    COLOR = 8
  };
  // [triflow] Triangle::border[3]: per-edge border flag (replaces the
  // upstream vertex-only border tracking).
  struct Triangle { int v[3];double err[4];int deleted,dirty,attr;vec3f n;vec3f uvs[3];int material;int border[3]; };
  // [triflow] Vertex: adds custom_q (per-vertex user quadric), area
  // (accumulated 1-ring area for area-weighted / normalized error), and
  // tid (target-group id, -1 = ungrouped).
  struct Vertex { vec3f p;int tstart,tcount;SymetricMatrix q;SymetricMatrix custom_q;int border;double area; int tid; };
  struct Ref { int tid,tvertex; };
  std::vector<Triangle> triangles;
  std::vector<Vertex> vertices;
  std::vector<Ref> refs;
    std::string mtllib; //
    std::vector<std::string> materials; //
  
  // [triflow] fork globals
  double* custom_quadratics;       // user-supplied per-vertex 4x4 quadric (10 values), owned by caller
  int custom_quadratics_size;      // total double count (n_vertices * 10), 0 if unset
  bool has_target_vertex_ids = false; // set when target_vertex_ids[] is passed to simplify_mesh()
  double minimum_edge_length = -1.0;  // threshold for the intra-group short-edge collapse rule

  // [triflow][reserved] namespace-level tunables not wired to the Python API.
  // Kept for future extensibility. Do not remove without updating README.
  double border_weight = 1000.0;   // weight for the orthogonal border quadric
  int binary_search_steps = 0;     // binary-search iterations for flip fallback in calculate_error()
  bool override_quadratics = false;// future: replace-instead-of-add custom quadratic mode

  // Helper functions

  double vertex_error(SymetricMatrix q, double x, double y, double z);
  double calculate_error(int id_v1, int id_v2, vec3f &p_result);
  bool flipped(vec3f p,int i0,int i1,Vertex &v0,Vertex &v1,std::vector<int> &deleted);
  void update_uvs(int i0,const Vertex &v,const vec3f &p,std::vector<int> &deleted);
  void update_triangles(int i0,Vertex &v,std::vector<int> &deleted,int &deleted_triangles);
  void update_mesh(int iteration,void (*log)(char*, int));
  void compact_mesh();
  //
  // Main simplification function
  //
  // target_count  : target nr. of triangles
  // agressiveness : sharpness to increase the threshold.
  //                 5..8 are good numbers
  //                 more iterations yield higher quality
  //

  // [triflow] Fork adds three trailing parameters:
  //   custom_quadratics_param / custom_quadratics_size_param
  //       Per-vertex user quadric matrices (n_vertices * 10 doubles).
  //   target_vertex_ids / target_vertex_ids_size
  //       Per-vertex int group id (or -1). Drives the target-aware
  //       collapse rules in the main loop below.
  //   min_edge_length
  //       Threshold for the intra-group short-edge override.
  // These parameters are optional; passing nullptr / 0 / -1.0 yields
  // the standard (non-fork) behavior except for the always-on changes
  // marked elsewhere in this file (area-weighted error, non-manifold
  // pinch check, per-edge borders, etc.).
  void simplify_mesh(int target_count, int update_rate=5, double agressiveness=7,
                     void (*log)(char*, int)=NULL, int max_iterations=100, double alpha = 0.000000001,
                     int K = 3, bool lossless=false, double threshold_lossless = 0.0001,
                     bool preserve_border = false,
                     double* custom_quadratics_param = nullptr, int custom_quadratics_size_param = 0,
                     int* target_vertex_ids = nullptr, int target_vertex_ids_size = 0,
                     double min_edge_length = -1.0)
  {
    custom_quadratics = custom_quadratics_param;
    custom_quadratics_size = custom_quadratics_size_param;
    if (target_vertex_ids != nullptr) {
        has_target_vertex_ids = true;
        loopi(0, vertices.size()) {
            vertices[i].tid = target_vertex_ids[i];
        }
    }
    minimum_edge_length = min_edge_length;

    // init
    loopi(0,triangles.size())
        {
            triangles[i].deleted=0;
        }

    // main iteration loop
    int deleted_triangles=0;
    std::vector<int> deleted0,deleted1;
    int triangle_count=triangles.size();
    int previous_deleted_triangles = deleted_triangles;
    //int iteration = 0;
    //loop(iteration,0,100)
    for (int iteration = 0; iteration < max_iterations; iteration ++)
    {
      if(triangle_count-deleted_triangles<=target_count)break;

      // update mesh once in a while
      if((iteration%update_rate==0) || lossless)
      {
        update_mesh(iteration, log);
      }

      // clear dirty flag
      loopi(0,triangles.size()) triangles[i].dirty=0;

      //
      // All triangles with edges below the threshold will be removed
      //
      // The following numbers works well for most models.
      // If it does not, try to adjust the 3 parameters
      //
      double threshold = alpha*pow(double(iteration+K),agressiveness);
      if(lossless) threshold = threshold_lossless ;
      else threshold = min(threshold, threshold_lossless );

      if ((log) && (iteration%5==0)) {
        char message[128];
        snprintf(message, 127, "iteration %d - triangles %d threshold %g",iteration,triangle_count-deleted_triangles, threshold);
        log(message, 128);
      }

      // remove vertices & mark deleted triangles
      loopi(0,triangles.size())
      {
        Triangle &t=triangles[i];
        if(t.err[3]>threshold) continue;
        if(t.deleted) continue;
        if(t.dirty) continue;

        loopj(0,3)if(t.err[j]<threshold)
        {
          int i0=t.v[ j     ]; Vertex &v0 = vertices[i0];
          int i1=t.v[(j+1)%3]; Vertex &v1 = vertices[i1];

          // [triflow] target-group collapse direction rules: once a vertex
          // is tagged as belonging to a group, collapses can only flow
          // *towards* a seed, never away from one.
          if (has_target_vertex_ids && v0.tid != -1 && v1.tid == -1) {
            continue; // only allow collapsing towards skeleton
          }
          if (has_target_vertex_ids && v0.tid != i0 && v1.tid == i1) {
            continue; // only allow collapsing towards seed
          }

          // Border check //Added preserve_border method from issue 14
          if(preserve_border){
            if (v0.border || v1.border) continue; // should keep border vertices
          }
          // [triflow] edge-level border rule: allow collapse along a border
          // edge, but reject when both endpoints are borders yet the shared
          // edge is not (i.e. they're on two different border components).
          if (v0.border && v1.border && !t.border[j]) continue;

          // [triflow] target-group split check: reject a collapse between
          // two different target groups unless the edge is short enough,
          // and also reject any collapse that would disconnect v1's
          // same-group ring neighbours (detected via union-find below).
          if (has_target_vertex_ids && v0.tid != v1.tid && v0.tid != -1 && v1.tid != -1) {
              if ((v0.p - v1.p).length() > minimum_edge_length) {
                  continue;
              }

              int target_tid = v1.tid;
              int max_neighbors = v1.tcount * 2;
              
              int* group_neighbors = new int[max_neighbors];
              int* parent = new int[max_neighbors]; 
              int group_neighbor_count = 0;

              auto get_or_create_local_id = [&](int global_id) {
                  for (int n_idx = 0; n_idx < group_neighbor_count; n_idx++) 
                      if (group_neighbors[n_idx] == global_id) return n_idx;
                  
                  group_neighbors[group_neighbor_count] = global_id;
                  parent[group_neighbor_count] = group_neighbor_count;
                  return group_neighbor_count++;
              };

              auto find_root = [&](int node, auto& self) -> int {
                  if (parent[node] == node) return node;
                  return parent[node] = self(parent[node], self);
              };

              for (int m = 0; m < v1.tcount; m++) {
                  Triangle &t_ring = triangles[refs[v1.tstart + m].tid];
                  if (t_ring.deleted) continue;

                  int va = -1, vb = -1;
                  for (int n = 0; n < 3; n++) {
                      if (t_ring.v[n] != i1) {
                          if (va == -1) va = t_ring.v[n];
                          else vb = t_ring.v[n];
                      }
                  }
                  
                  int la = -1, lb = -1;
                  // Register all neighbors belonging to the target group
                  if (va != -1 && vertices[va].tid == target_tid) la = get_or_create_local_id(va);
                  if (vb != -1 && vertices[vb].tid == target_tid) lb = get_or_create_local_id(vb);

                  // If both are in target_tid, union them
                  if (la != -1 && lb != -1) {
                      int root_a = find_root(la, find_root);
                      int root_b = find_root(lb, find_root);
                      if (root_a != root_b) parent[root_a] = root_b;
                  }
              }

              bool split_detected = false;
              if (group_neighbor_count > 1) {
                  int final_root = find_root(0, find_root);
                  for (int m = 1; m < group_neighbor_count; m++) {
                      if (find_root(m, find_root) != final_root) {
                          split_detected = true;
                          break;
                      }
                  }
              }

              delete[] group_neighbors;
              delete[] parent;

              if (split_detected) {
                  continue;
              }
          }

          // [triflow] non-manifold pinch check: reject a collapse whose
          // resulting vertex would have two disjoint triangle fans sharing
          // only the collapsed vertex (i.e. a pinched / non-manifold point).
          // Implemented by walking v0's ring, flagging every neighbour as
          // "safe" if it appears on a triangle shared with v1, and then
          // checking v1's ring for any shared neighbour that was not
          // flagged safe.
          if (true) {
              int max_v0_neighbors = v0.tcount * 2;
              int* v0_neighbors = new int[max_v0_neighbors];
              bool* is_safe = new bool[max_v0_neighbors];
              int v0_n_count = 0;

              // Helper: Finds the local index of a vertex
              auto get_neighbor_index = [&](int vid) -> int {
                  for (int n_idx = 0; n_idx < v0_n_count; n_idx++) {
                      if (v0_neighbors[n_idx] == vid) return n_idx;
                  }
                  return -1;
              };

              // Helper: Adds or updates a neighbor's safety status
              auto update_neighbor = [&](int vid, bool safe) {
                  if (vid == i0 || vid == i1) return;
                  int ex = get_neighbor_index(vid);
                  if (ex == -1) {
                      if (v0_n_count < max_v0_neighbors) {
                          v0_neighbors[v0_n_count] = vid;
                          is_safe[v0_n_count] = safe;
                          v0_n_count++;
                      }
                  } else {
                      // If it's part of ANY shared triangle, it's safe
                      if (safe) is_safe[ex] = true;
                  }
              };

              // 1. Pass through v0's triangles to build the "Link" map
              for (int m = 0; m < v0.tcount; m++) {
                  Triangle &t_v0 = triangles[refs[v0.tstart + m].tid];
                  if (t_v0.deleted) continue;

                  bool shares_with_v1 = false;
                  for (int n = 0; n < 3; n++) {
                      if (t_v0.v[n] == i1) { shares_with_v1 = true; break; }
                  }

                  for (int n = 0; n < 3; n++) update_neighbor(t_v0.v[n], shares_with_v1);
              }

              // 2. Check v1's neighbors against the list
              bool pinch_detected = false;
              for (int m = 0; m < v1.tcount; m++) {
                  Triangle &t_v1 = triangles[refs[v1.tstart + m].tid];
                  if (t_v1.deleted) continue;

                  for (int n = 0; n < 3; n++) {
                      int vid = t_v1.v[n];
                      if (vid == i0 || vid == i1) continue;

                      int ex = get_neighbor_index(vid);
                      // If vid is a neighbor of both, but NOT marked safe, it's a non-manifold pinch
                      if (ex != -1 && !is_safe[ex]) {
                          pinch_detected = true;
                          break;
                      }
                  }
                  if (pinch_detected) break;
              }

              delete[] v0_neighbors;
              delete[] is_safe;

              if (pinch_detected) {
                  continue;
              }
          }

          // Compute vertex to collapse to
          vec3f p;
          calculate_error(i0,i1,p);
          deleted0.resize(v0.tcount); // normals temporarily
          deleted1.resize(v1.tcount); // normals temporarily

          // Call flipped() for its side effect of populating deleted0/deleted1.
          // Flip rejection is intentionally disabled; the optimal-position
          // safety fallback in calculate_error handles runaway points instead.
          flipped(p,i0,i1,v0,v1,deleted0);
          flipped(p,i1,i0,v1,v0,deleted1);

          if ( (t.attr & TEXCOORD) == TEXCOORD  )
          {
            update_uvs(i0,v0,p,deleted0);
            update_uvs(i0,v1,p,deleted1);
          }

          // not flipped, so remove edge
          v0.p=p;
          v0.q=v1.q+v0.q;
          if ( custom_quadratics != nullptr && custom_quadratics_size > 0 )
          {
              // v0.custom_q = v0.custom_q[0] > v1.custom_q[0] ? v0.custom_q : v1.custom_q;
              v0.custom_q = v0.custom_q + v1.custom_q;
          }
          v0.area = v0.area + v1.area;
          // [triflow] if v1 was a target-group seed, reassign every member
          // of its group to i0 so the group's identity follows the seed.
          if (v1.tid == i1){
            loopk(0, vertices.size()){
              if (vertices[k].tid == i1){
                vertices[k].tid = i0;
              }
            }
          }
          // [triflow] propagate edge-level border flags on collapse: any
          // triangle of v0 that inherits a border neighbour from v1's
          // ring needs its matching edge marked as a border.
          // update border
          bool was_v0_border = v0.border;
          bool was_v1_border = v1.border;
          if (was_v1_border) v0.border = true;
          if (!was_v0_border && was_v1_border) {
            int max_triangles = v1.tcount * 2;
            int* anchor_vertices = new int[max_triangles];
            int anchor_count = 0;

            for (int k = 0; k < v1.tcount; k++) {
              Triangle &t = triangles[refs[v1.tstart + k].tid];
              if (t.deleted || !deleted1[k]) continue;
              for (int m = 0; m < 3; m++) {
                if (!t.border[m]) continue;
                int va = t.v[m];
                int vb = t.v[(m + 1) % 3];
                int v_other = (va == i1) ? vb : (vb == i1) ? va : -1;
                if (v_other == -1 || v_other == i0) continue;
                // add to anchor list
                anchor_vertices[anchor_count++] = v_other;
              }
            }

            for (int k = 0; k < v0.tcount; k++) {
              Triangle &t = triangles[refs[v0.tstart + k].tid];
              if (t.deleted || deleted0[k]) continue;
              for (int m = 0; m < 3; m++) {
                int va = t.v[m];
                int vb = t.v[(m + 1) % 3];
                int v_other = (va == i0) ? vb : (vb == i0) ? va : -1;
                if (v_other == -1) continue;
                for (int n = 0; n < anchor_count; n++) {
                  if (v_other == anchor_vertices[n]) {
                    t.border[m] = true;
                    break;
                  }
                }
              }
            }
          }

          int tstart=refs.size();

          update_triangles(i0,v0,deleted0,deleted_triangles);
          update_triangles(i0,v1,deleted1,deleted_triangles);

          int tcount=refs.size()-tstart;

          if(tcount<=v0.tcount)
          {
            // save ram
            if(tcount)memcpy(&refs[v0.tstart],&refs[tstart],tcount*sizeof(Ref));
          }
          else
            // append
            v0.tstart=tstart;

          v0.tcount=tcount;
          break;
        }
        // done?
        if (!lossless && (triangle_count-deleted_triangles<=target_count)){break;}

        // if (lossless) deleted_triangles = 0;
      }

      // [triflow] termination: upstream only broke on target_count or,
      // in lossless mode, on deleted_triangles<=0 (zeroed every iter).
      // This fork also breaks when an iteration made no progress and
      // the threshold has already saturated at threshold_lossless --
      // i.e. further iterations cannot collapse anything new.
      if (lossless && (previous_deleted_triangles==deleted_triangles)){ break;
      } else if (!lossless && (triangle_count-deleted_triangles<=target_count)){break;
      } else if (!lossless && (previous_deleted_triangles==deleted_triangles) && threshold >= threshold_lossless ) {
        break;
      }

      previous_deleted_triangles = deleted_triangles;
    }
    // clean up mesh
    compact_mesh();
  } //simplify_mesh()

  void simplify_mesh_lossless(void (*log)(char*, int)=NULL, double epsilon=1e-3, int max_iterations = 9999, bool preserve_border = false)
  {
    // init
    loopi(0,triangles.size())
        {
            triangles[i].deleted=0;
        }
    // main iteration loop
    int deleted_triangles=0;
    std::vector<int> deleted0,deleted1;
    int triangle_count=triangles.size();
    //int iteration = 0;
    //loop(iteration,0,100)
    for (int iteration = 0; iteration < max_iterations; iteration ++)
    {
      // update mesh constantly
      update_mesh(iteration, log);
      // clear dirty flag
      loopi(0,triangles.size()) triangles[i].dirty=0;
      //
      // All triangles with edges below the threshold will be removed
      //
      // The following numbers works well for most models.
      // If it does not, try to adjust the 3 parameters
      //
      double threshold = epsilon; //1.0E-3 EPS;
      if (log) {
        char message[128];
        snprintf(message, 127, "lossless iteration %d\n", iteration);
        log(message, 128);
      }

      // remove vertices & mark deleted triangles
      loopi(0,triangles.size())
      {
        Triangle &t=triangles[i];
        if(t.err[3]>threshold) continue;
        if(t.deleted) continue;
        if(t.dirty) continue;

        loopj(0,3)if(t.err[j]<threshold)
        {
          int i0=t.v[ j     ]; Vertex &v0 = vertices[i0];
          int i1=t.v[(j+1)%3]; Vertex &v1 = vertices[i1];

          // Border check //Added preserve_border method from issue 14 for lossless
          if(preserve_border){
            if (v0.border || v1.border) continue; // should keep border vertices
          }
          else
            if (v0.border != v1.border)  continue; // base behaviour

          // Compute vertex to collapse to
          vec3f p;
          calculate_error(i0,i1,p);

          deleted0.resize(v0.tcount); // normals temporarily
          deleted1.resize(v1.tcount); // normals temporarily

          // don't remove if flipped
          if( flipped(p,i0,i1,v0,v1,deleted0) ) continue;
          if( flipped(p,i1,i0,v1,v0,deleted1) ) continue;

          if ( (t.attr & TEXCOORD) == TEXCOORD )
          {
            update_uvs(i0,v0,p,deleted0);
            update_uvs(i0,v1,p,deleted1);
          }

          // not flipped, so remove edge
          v0.p=p;
          v0.q=v1.q+v0.q;
          int tstart=refs.size();

          update_triangles(i0,v0,deleted0,deleted_triangles);
          update_triangles(i0,v1,deleted1,deleted_triangles);

          int tcount=refs.size()-tstart;

          if(tcount<=v0.tcount)
          {
            // save ram
            if(tcount)memcpy(&refs[v0.tstart],&refs[tstart],tcount*sizeof(Ref));
          }
          else
            // append
            v0.tstart=tstart;

          v0.tcount=tcount;
          break;
        }
      }
      if(deleted_triangles<=0)break;
      deleted_triangles=0;
    } //for each iteration
    // clean up mesh
    compact_mesh();
  } //simplify_mesh_lossless()


  // Check if a triangle flips when this edge is removed

  // [triflow] No longer early-returns on the first bad triangle. Upstream
  // returned true as soon as a flip was detected, leaving deleted[] only
  // partially populated. This fork always scans the full 1-ring so
  // deleted[] is fully filled -- required by the border-propagation code
  // in the main simplify_mesh() collapse block, which reads deleted[k]
  // for every k in v1's ring.
  bool flipped(vec3f p,int i0,int i1,Vertex &v0,Vertex &v1,std::vector<int> &deleted)
  {
    bool is_flipped=false;
    loopk(0,v0.tcount)
    {
      Triangle &t=triangles[refs[v0.tstart+k].tid]; // for each adjacent triangle of v0
      if(t.deleted)continue;

      int s=refs[v0.tstart+k].tvertex;
      int id1=t.v[(s+1)%3];
      int id2=t.v[(s+2)%3];

      if(id1==i1 || id2==i1) // delete ?
      {
        deleted[k]=1;
        continue;
      }else{
        deleted[k]=0;
      }

      vec3f d1 = vertices[id1].p-p; d1.normalize();
      vec3f d2 = vertices[id2].p-p; d2.normalize();
      if(fabs(d1.dot(d2))>0.999) is_flipped=true;
      vec3f n;
      n.cross(d1,d2);
      n.normalize();
      // deleted[k]=0;
      if(n.dot(t.n)<0.2) is_flipped=true;
    }
    return is_flipped;
  }

    // update_uvs

  void update_uvs(int i0,const Vertex &v,const vec3f &p,std::vector<int> &deleted)
  {
    loopk(0,v.tcount)
    {
      Ref &r=refs[v.tstart+k];
      Triangle &t=triangles[r.tid];
      if(t.deleted)continue;
      if(deleted[k])continue;
      vec3f p1=vertices[t.v[0]].p;
      vec3f p2=vertices[t.v[1]].p;
      vec3f p3=vertices[t.v[2]].p;
      t.uvs[r.tvertex] = interpolate(p,p1,p2,p3,t.uvs);
    }
  }

  // Update triangle connections and edge error after a edge is collapsed

  void update_triangles(int i0,Vertex &v,std::vector<int> &deleted,int &deleted_triangles)
  {
    vec3f p;
    loopk(0,v.tcount)
    {
      Ref &r=refs[v.tstart+k];
      Triangle &t=triangles[r.tid];
      if(t.deleted)continue;
      if(deleted[k])
      {
        t.deleted=1;
        deleted_triangles++;
        continue;
      }
      t.v[r.tvertex]=i0;
      t.dirty=1;
      t.err[0]=calculate_error(t.v[0],t.v[1],p);
      t.err[1]=calculate_error(t.v[1],t.v[2],p);
      t.err[2]=calculate_error(t.v[2],t.v[0],p);
      t.err[3]=min(t.err[0],min(t.err[1],t.err[2]));
      refs.push_back(r);
    }
  }

  // compact triangles, compute edge error and build reference list

  void update_mesh(int iteration, void (*log)(char*, int)=NULL)
  {
    if(iteration>0) // compact triangles
    {
      int dst=0;
      loopi(0,triangles.size())
      if(!triangles[i].deleted)
      {
        triangles[dst++]=triangles[i];
      }
      triangles.resize(dst);
    }

    // Init Reference ID list
    if ((log) && (iteration%5==0)) {
      char message[128];
      snprintf(message, 127, "Init Reference ID list");
      log(message, 128);
    }
    loopi(0,vertices.size())
    {
      vertices[i].tstart=0;
      vertices[i].tcount=0;
    }
    loopi(0,triangles.size())
    {
      Triangle &t=triangles[i];
      loopj(0,3) vertices[t.v[j]].tcount++;
    }
    int tstart=0;
    loopi(0,vertices.size())
    {
      Vertex &v=vertices[i];
      v.tstart=tstart;
      tstart+=v.tcount;
      v.tcount=0;
    }

    // Write References
    if ((log) && (iteration%5==0)) {
      char message[128];
      snprintf(message, 127, "Write References");
      log(message, 128);
    }
    refs.resize(triangles.size()*3);
    loopi(0,triangles.size())
    {
      Triangle &t=triangles[i];
      loopj(0,3)
      {
        Vertex &v=vertices[t.v[j]];
        refs[v.tstart+v.tcount].tid=i;
        refs[v.tstart+v.tcount].tvertex=j;
        v.tcount++;
      }
    }

    // Identify boundary : vertices[].border=0,1
    if ((log) && (iteration%5==0)) {
      char message[128];
      snprintf(message, 127, "Identify boundary");
      log(message, 128);
    }
    if( iteration == 0 )
    {
      std::vector<int> vcount,vids;

      // [triflow] also reset per-edge border flags Triangle::border[j],
      // not just per-vertex Vertex::border as upstream did.
      loopi(0,vertices.size())
        vertices[i].border=0;
      loopi(0,triangles.size())
      {
        Triangle &t=triangles[i];
        loopj(0,3)
        {
          t.border[j]=0;
        }
      }

      loopi(0,vertices.size())
      {
        Vertex &v=vertices[i]; // for each vertex
        vcount.clear();
        vids.clear();
        loopj(0,v.tcount) 
        {
          int k=refs[v.tstart+j].tid;
          Triangle &t=triangles[k]; // for each adjacent triangle
          loopk(0,3)
          {
            int ofs=0,id=t.v[k]; // for each vertex of triangle
            while(ofs<vcount.size()) // ofs: index in vcount/vids of the current vertex
            {
              if(vids[ofs]==id)break;
              ofs++;
            }
            if(ofs==vcount.size())
            {
              vcount.push_back(1);
              vids.push_back(id);
            }
            else
              vcount[ofs]++;
          }
        }
        loopj(0,vcount.size()) if(vcount[j]==1)
          vertices[vids[j]].border=1;
        loopj(0,v.tcount) 
        {
          int k=refs[v.tstart+j].tid;
          Triangle &t=triangles[k]; // for each adjacent triangle
          loopk(0,3)
          {
            int id1=t.v[k];
            int id2=t.v[(k+1)%3];
            int other_id;
            
            if ( id1 == i ) other_id = id2;
            else if ( id2 == i ) other_id = id1;
            else continue;

            int ofs=0;
            while(ofs<vcount.size()) // ofs: index in vcount/vids of the current vertex
            {
              if(vids[ofs]==other_id)break;
              ofs++;
            }

            if(ofs<vcount.size() && vcount[ofs]==1) t.border[k]=1;
          }
        }
      }
    }

    //
    // Init Quadrics by Plane & Edge Errors
    //
    // required at the beginning ( iteration == 0 )
    // recomputing during the simplification is not required,
    // but mostly improves the result for closed meshes
    //
    if( iteration == 0 )
    {
      if ((log) && (iteration%5==0)) {
        char message[128];
        snprintf(message, 127, "Init Quadrics");
        log(message, 128);
      }
      loopi(0,vertices.size()){
        vertices[i].q=SymetricMatrix(0.0);
        vertices[i].area = 0.0;
      }
      loopi(0,triangles.size())
      {
        Triangle &t=triangles[i];
        vec3f n,p[3];
        loopj(0,3) p[j]=vertices[t.v[j]].p;
        n.cross(p[1]-p[0],p[2]-p[0]);
        double area = n.length() * 0.5;
        n.normalize();
        t.n=n;
        loopj(0,3) {
          // [triflow] accumulate per-vertex 1-ring area, used later for
          // area-weighting quadrics and for normalizing reported error.
          vertices[t.v[j]].area += area;
          // distance to plane
          vertices[t.v[j]].q = vertices[t.v[j]].q+SymetricMatrix(n.x,n.y,n.z,-n.dot(p[0]));
          // [triflow] for border edges, add a second quadric measuring
          // distance to the plane *orthogonal* to the triangle that
          // contains the border edge. Scaled by border_weight to pin
          // border vertices in place.
          if ( border_weight > 0.0 && t.border[j] )
          {
            vec3f edge = p[(j + 1) % 3] - p[j];
            edge.normalize();
            
            vec3f nb; // Boundary normal
            nb.cross(t.n, edge);
            nb.normalize();
            float d = -nb.dot(p[j]);

            // Add orthogonal plane quadric to both vertices of the edge
            SymetricMatrix q_border(nb.x, nb.y, nb.z, d);
            vertices[t.v[j]].q = vertices[t.v[j]].q + (q_border * border_weight);
            vertices[t.v[(j + 1) % 3]].q = vertices[t.v[(j + 1) % 3]].q + (q_border * border_weight);
          }
        }
      }

      // [triflow] apply user-supplied custom quadratics. Stored on
      // Vertex::custom_q separately from Vertex::q so that the reported
      // geometric error in calculate_error() can still be computed
      // against q (geometry-only) even though the collapse *position*
      // is solved using q + custom_q.
      if ((log) && (iteration%5==0)) {
        char message[128];
        snprintf(message, 127, "apply custom quadratics if provided");
        log(message, 128);
      }
      if (custom_quadratics != nullptr && custom_quadratics_size > 0)
      {
        loopi(0, vertices.size())
        {
          // Ensure we don't go out of bounds
          // custom_quadratics should be 10*n_vertices (10 values per vertex for symmetric matrix)
          int offset = i * 10;
          if (offset + 9 < custom_quadratics_size)
          {
            // Create symmetric matrix from the 10 values
            // Order: m11, m12, m13, m14, m22, m23, m24, m33, m34, m44
            SymetricMatrix custom_q(
                custom_quadratics[offset + 0],  // m11
                custom_quadratics[offset + 1],  // m12
                custom_quadratics[offset + 2],  // m13
                custom_quadratics[offset + 3],  // m14
                custom_quadratics[offset + 4],  // m22
                custom_quadratics[offset + 5],  // m23
                custom_quadratics[offset + 6],  // m24
                custom_quadratics[offset + 7],  // m33
                custom_quadratics[offset + 8],  // m34
                custom_quadratics[offset + 9]   // m44
            );
            // if (override_quadratics) vertices[i].q = custom_q;
            // else vertices[i].q = vertices[i].q + custom_q;
            vertices[i].custom_q = custom_q;
          }
        }
      }

      // [triflow] area-weight the quadrics so that larger triangles
      // dominate the collapse error. Combined with area-normalized
      // error in calculate_error(), this yields a scale-invariant
      // simplification metric.
      loopi(0, vertices.size()) {
        double area = vertices[i].area;
        if (area > 0.0) {
          vertices[i].q *= area;
          if ( custom_quadratics != nullptr && custom_quadratics_size > 0 )
          {
              vertices[i].custom_q *= area;
          }
        }
      }

      if ((log) && (iteration%5==0)) {
        char message[128];
        snprintf(message, 127, "Calc Edge Error");
        log(message, 128);
      }
      loopi(0,triangles.size())
      {
        // Calc Edge Error
        Triangle &t=triangles[i];vec3f p;
        loopj(0,3) t.err[j]=calculate_error(t.v[j],t.v[(j+1)%3],p);
        t.err[3]=min(t.err[0],min(t.err[1],t.err[2]));
      }
    }

  }

  // Finally compact mesh before exiting

  void compact_mesh()
  {
    int dst=0;
    loopi(0,vertices.size())
    {
      vertices[i].tcount=0;
    }
    loopi(0,triangles.size())
    if(!triangles[i].deleted)
    {
      Triangle &t=triangles[i];
      triangles[dst++]=t;
      loopj(0,3)vertices[t.v[j]].tcount=1;
    }
    triangles.resize(dst);
    dst=0;
    loopi(0,vertices.size())
    if(vertices[i].tcount)
    {
      vertices[i].tstart=dst;
      vertices[dst].p=vertices[i].p;
      dst++;
    }
    loopi(0,triangles.size())
    {
      Triangle &t=triangles[i];
      loopj(0,3)t.v[j]=vertices[t.v[j]].tstart;
    }
    vertices.resize(dst);
  }

  // Error between vertex and Quadric

  double vertex_error(SymetricMatrix q, double x, double y, double z)
  {
    return   q[0]*x*x + 2*q[1]*x*y + 2*q[2]*x*z + 2*q[3]*x + q[4]*y*y
         + 2*q[5]*y*z + 2*q[6]*y + q[7]*z*z + 2*q[8]*z + q[9];
  }

  // Error for one edge

  double calculate_error(int id_v1, int id_v2, vec3f &p_result)
  {
    // compute interpolated vertex

    // [triflow] q_geom holds geometry-only quadric for error reporting;
    // q is the solver quadric used to pick the collapse position and
    // includes any user-supplied custom quadratic bias.
    SymetricMatrix q_geom = vertices[id_v1].q + vertices[id_v2].q;
    SymetricMatrix q = vertices[id_v1].q + vertices[id_v2].q;
    if ( custom_quadratics != nullptr && custom_quadratics_size > 0 )
    {
        q += vertices[id_v1].custom_q + vertices[id_v2].custom_q;
    }
    bool   border = vertices[id_v1].border & vertices[id_v2].border;
    double error=0;
    double det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7);
    // [triflow] loosened fallback condition: upstream only fell back on
    // singular q; we also fall back on near-singular q (det < 1e-6) and
    // on optimal points that land implausibly far from the edge midpoint
    // (see the opt_length check below). The binary_search_steps path
    // is reserved but currently disabled (binary_search_steps == 0).
    bool fallback = det < 1e-6 ;
    bool is_flipped = false;
    if ( !fallback ) // det != 0 && !border
    {
      // q_delta is invertible
      p_result.x = -1/det*(q.det(1, 2, 3, 4, 5, 6, 5, 7 , 8));  // vx = A41/det(q_delta)
      p_result.y =  1/det*(q.det(0, 2, 3, 1, 5, 6, 2, 7 , 8));  // vy = A42/det(q_delta)
      p_result.z = -1/det*(q.det(0, 1, 3, 1, 4, 6, 2, 5,  8));  // vz = A43/det(q_delta)
      error = vertex_error(q, p_result.x, p_result.y, p_result.z);

      vec3f p1 = vertices[id_v1].p;
      vec3f p2 = vertices[id_v2].p;
      vec3f p_mid = (p1 + p2) / 2.0f;
      double edge_length = (p1 - p2).length();
      double opt_length = (p_result - p_mid).length();
      if ( opt_length > edge_length * 2.0 ) {
        // if the optimal point is too far from the edge, it's likely to cause flips or other issues, so we fallback to the midpoint
        fallback = true;
      }
    }
    if ( fallback )
    {
      // 1. Find the initial "Safe" point among v1, v2, and midpoint
      vec3f p1 = vertices[id_v1].p;
      vec3f p2 = vertices[id_v2].p;
      vec3f p_safe = (p1 + p2) / 2.0f; 
      
      double error1 = vertex_error(q, p1.x, p1.y, p1.z);
      double error2 = vertex_error(q, p2.x, p2.y, p2.z);
      double error3 = vertex_error(q, p_safe.x, p_safe.y, p_safe.z);
      
      double best_safe_error = error3;
      if (error1 < best_safe_error) { p_safe = p1; best_safe_error = error1; }
      if (error2 < best_safe_error) { p_safe = p2; best_safe_error = error2; }

      if (is_flipped) { // there was an optimal point but it caused a flip

        // 2. Prepare for Binary Search
        // We know p_result (the optimal) flips, and p_safe is our fallback.
        vec3f p_opt = p_result; 
        Vertex &v0 = vertices[id_v1];
        Vertex &v1 = vertices[id_v2];
        std::vector<int> deleted0(v0.tcount), deleted1(v1.tcount);

        // 3. Binary Search Iterations
        loopi(0, binary_search_steps){
          vec3f p_test = (p_opt + p_safe) / 2.0f;
          
          // Check if this interpolated point causes a flip
          if (flipped(p_test, id_v1, id_v2, v0, v1, deleted0) || 
              flipped(p_test, id_v2, id_v1, v1, v0, deleted1)) 
          {
            // Still flipping: move p_opt closer to the safe point
            p_opt = p_test;
          } 
          else 
          {
            // Not flipping: move p_safe closer to the optimal point
            p_safe = p_test;
          }
        }

        // 4. Set final result to the best non-flipping point found
        p_result = p_safe;
        error = vertex_error(q, p_result.x, p_result.y, p_result.z);
      }else{
        error = best_safe_error;
        p_result = p_safe;
      }
    }

    // [triflow] reported error is geometry-only (q_geom, not q) and
    // normalized by the sum of endpoint 1-ring areas, yielding a
    // scale-invariant metric that is comparable across meshes of
    // very different absolute sizes.
    double raw_geom_error = vertex_error(q_geom, p_result.x, p_result.y, p_result.z);
    double total_area = vertices[id_v1].area + vertices[id_v2].area;
    double normalized_error = raw_geom_error / (total_area + 1e-6); // avoid division by zero

    return normalized_error;
  }

  char *trimwhitespace(char *str)
  {
    char *end;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0)  // All spaces?
    return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
  }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////// Not Useful for Trimesh wrapper, we rather need to directly assign vertices, faces colors etc. ////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //Option : Load OBJ
  void load_obj(const char* filename, bool process_uv=false){
    vertices.clear();
    triangles.clear();
    //printf ( "Loading Objects %s ... \n",filename);
    FILE* fn;
    if(filename==NULL)    return ;
    if((char)filename[0]==0)  return ;
    if ((fn = fopen(filename, "rb")) == NULL)
    {
      printf ( "File %s not found!\n" ,filename );
      return;
    }
    char line[1000];
    memset ( line,0,1000 );
    int vertex_cnt = 0;
    int material = -1;
    std::map<std::string, int> material_map;
    std::vector<vec3f> uvs;
    std::vector<std::vector<int> > uvMap;

    while(fgets( line, 1000, fn ) != NULL)
    {
      Vertex v;
      vec3f uv;

      if (strncmp(line, "mtllib", 6) == 0)  ///not important
      { ///not important
        mtllib = trimwhitespace(&line[7]);  ///not important
      } ///not important
      if (strncmp(line, "usemtl", 6) == 0)  ///not important
      { ///not important
        std::string usemtl = trimwhitespace(&line[7]);  ///not important
        if (material_map.find(usemtl) == material_map.end())  ///not important
        { ///not important
          material_map[usemtl] = materials.size();  ///not important
          materials.push_back(usemtl);  ///not important
        } ///not important
        material = material_map[usemtl];  ///not important
      } ///not important

      if ( line[0] == 'v' && line[1] == 't' )
      {
        if ( line[2] == ' ' )
        if(sscanf(line,"vt %lf %lf",
          &uv.x,&uv.y)==2)
        {
          uv.z = 0;
          uvs.push_back(uv);
        } else
        if(sscanf(line,"vt %lf %lf %lf",
          &uv.x,&uv.y,&uv.z)==3)
        {
          uvs.push_back(uv);
        }
      }
      else if ( line[0] == 'v' )
      {
        if ( line[1] == ' ' )
        if(sscanf(line,"v %lf %lf %lf",
          &v.p.x, &v.p.y, &v.p.z)==3)
        {
          vertices.push_back(v);
        }
      }
      int integers[9];
      if ( line[0] == 'f' )
      {
        Triangle t;
        bool tri_ok = false;
                bool has_uv = false;

        if(sscanf(line,"f %d %d %d",
          &integers[0],&integers[1],&integers[2])==3)
        {
          tri_ok = true;
        }else
        if(sscanf(line,"f %d// %d// %d//",
          &integers[0],&integers[1],&integers[2])==3)
        {
          tri_ok = true;
        }else
        if(sscanf(line,"f %d//%d %d//%d %d//%d",
          &integers[0],&integers[3],
          &integers[1],&integers[4],
          &integers[2],&integers[5])==6)
        {
          tri_ok = true;
        }else
        if(sscanf(line,"f %d/%d/%d %d/%d/%d %d/%d/%d",
          &integers[0],&integers[6],&integers[3],
          &integers[1],&integers[7],&integers[4],
          &integers[2],&integers[8],&integers[5])==9)
        {
          tri_ok = true;
          has_uv = true;
        }
        else
        {
          printf("unrecognized sequence\n");
          printf("%s\n",line);
          while(1);
        }
        if ( tri_ok )
        {
          t.v[0] = integers[0]-1-vertex_cnt;
          t.v[1] = integers[1]-1-vertex_cnt;
          t.v[2] = integers[2]-1-vertex_cnt;
          t.attr = 0;

          if ( process_uv && has_uv )
          {
            std::vector<int> indices;
            indices.push_back(integers[6]-1-vertex_cnt);
            indices.push_back(integers[7]-1-vertex_cnt);
            indices.push_back(integers[8]-1-vertex_cnt);
            uvMap.push_back(indices);
            t.attr |= TEXCOORD;
          }

          t.material = material;
          //geo.triangles.push_back ( tri );
          triangles.push_back(t);
          //state_before = state;
          //state ='f';
        }
      }
    }

    if ( process_uv && uvs.size() )
    {
      loopi(0,triangles.size())
      {
        loopj(0,3)
        triangles[i].uvs[j] = uvs[uvMap[i][j]];
      }
    }

    fclose(fn);

    //printf("load_obj: vertices = %lu, triangles = %lu, uvs = %lu\n", vertices.size(), triangles.size(), uvs.size() );
  } // load_obj()



  void setMeshFromExt(std::vector< std::vector<double> > verts, std::vector< std::vector<int> > faces){
    vertices.clear();
    triangles.clear();

    int N_faces = faces.size();
    int N_vertices = verts.size();
    loopi(0,N_vertices)
    {
      Vertex v;
      v.p.x = verts[i][0];
      v.p.y = verts[i][1];
      v.p.z = verts[i][2];
      vertices.push_back(v);
    }
    loopi(0,N_faces)
    {
      Triangle t;
      t.v[0] = faces[i][0];
      t.v[1] = faces[i][1];
      t.v[2] = faces[i][2];
      t.attr = 0 ;
      t.material = -1 ;
      triangles.push_back(t);
    }
  }

  std::vector< std::vector<double> > getVertices(){
    std::vector< std::vector<double> > verts;
    int N_vertices = vertices.size();
    loopi(0,N_vertices)
    {
      std::vector<double> v;
      v.push_back(vertices[ i].p.x);
      v.push_back(vertices[ i].p.y);
      v.push_back(vertices[ i].p.z);
      verts.push_back(v);
    }
    return verts ;
  }

  std::vector< std::vector<int> > getFaces(){
    std::vector< std::vector<int> > faces;
    int N_faces = triangles.size();
    loopi(0,N_faces)
    {
      std::vector<int> t;
      loopj(0,3)
      {
        t.push_back(triangles[ i ].v[ j ]);
      }
    faces.push_back(t);
    }
    return faces ;
  }


  std::vector< std::vector<double> > getNormals(){
    std::vector< std::vector<double> > normals;
    int N_faces = triangles.size();
    loopi(0,N_faces)
    {
      std::vector<double> n;
      n.push_back(triangles[ i ].n.x);
      n.push_back(triangles[ i ].n.y);
      n.push_back(triangles[ i ].n.z);
      normals.push_back(n);
    }
    return normals ;
  }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////// Not Useful for Trimesh wrapper, we rather need to directly return vertices, faces colors etc. ////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void write_obj(const char* filename)
  {
    FILE *file=fopen(filename, "w");
    int cur_material = -1;
    bool has_uv = (triangles.size() && (triangles[0].attr & TEXCOORD) == TEXCOORD);

    if (!file)
    {
      printf("write_obj: can't write data file \"%s\".\n", filename);
      exit(0);
    }
    if (!mtllib.empty())
    {
      fprintf(file, "mtllib %s\n", mtllib.c_str());
    }
    loopi(0,vertices.size())
    {
      //fprintf(file, "v %lf %lf %lf\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z);
      fprintf(file, "v %g %g %g\n", vertices[i].p.x,vertices[i].p.y,vertices[i].p.z); //more compact: remove trailing zeros
    }
    if (has_uv)
    {
      loopi(0,triangles.size()) if(!triangles[i].deleted)
      {
        fprintf(file, "vt %g %g\n", triangles[i].uvs[0].x, triangles[i].uvs[0].y);
        fprintf(file, "vt %g %g\n", triangles[i].uvs[1].x, triangles[i].uvs[1].y);
        fprintf(file, "vt %g %g\n", triangles[i].uvs[2].x, triangles[i].uvs[2].y);
      }
    }
    int uv = 1;
    loopi(0,triangles.size()) if(!triangles[i].deleted)
    {
      if (triangles[i].material != cur_material)
      {
        cur_material = triangles[i].material;
        fprintf(file, "usemtl %s\n", materials[triangles[i].material].c_str());
      }
      if (has_uv)
      {
        fprintf(file, "f %d/%d %d/%d %d/%d\n", triangles[i].v[0]+1, uv, triangles[i].v[1]+1, uv+1, triangles[i].v[2]+1, uv+2);
        uv += 3;
      }
      else
      {
        fprintf(file, "f %d %d %d\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1);
      }
      //fprintf(file, "f %d// %d// %d//\n", triangles[i].v[0]+1, triangles[i].v[1]+1, triangles[i].v[2]+1); //more compact: remove trailing zeros
    }
    fclose(file);
  }
};
///////////////////////////////////////////
