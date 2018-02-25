#ifndef MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H
#define MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H

#ifndef MKL_BLAS
#define MKL_BLAS MKL_DOMAIN_BLAS
#endif

#define EIGEN_USE_MKL_ALL

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/LU>

#include "ceres/ceres.h"

#include <opencv2/opencv.hpp>

#include "basicmesh.h"
#include "common.h"
#include "constraints.h"
#include "costfunctions.h"
#include "multilinearmodel.h"
#include "parameters.h"
#include "singleimagereconstructor.hpp"
#include "statsutils.h"
#include "utils.hpp"

#include "OffscreenMeshVisualizer.h"

#include "AAM/aammodel.h"

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"

namespace fs = boost::filesystem;

using namespace Eigen;

namespace {
  struct PixelInfo {
    PixelInfo() : fidx(-1) {}
    PixelInfo(int fidx, glm::vec3 bcoords) : fidx(fidx), bcoords(bcoords) {}

    int fidx;           // trinagle index
    glm::vec3 bcoords;  // bary centric coordinates
  };

  inline void encode_index(int idx, unsigned char& r, unsigned char& g, unsigned char& b) {
    r = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    g = static_cast<unsigned char>(idx & 0xff); idx >>= 8;
    b = static_cast<unsigned char>(idx & 0xff);
  }

  inline int decode_index(unsigned char r, unsigned char g, unsigned char b, int& idx) {
    idx = b; idx <<= 8; idx |= g; idx <<= 8; idx |= r;
    return idx;
  }

  template <typename T>
  T clamp(T val, T lower, T upper) {
    return std::max(lower, std::min(upper, val));
  }

  inline glm::dvec3 bilinear_sample(const QImage& img, double x, double y) {
    int x0 = floor(x), x1 = x0 + 1;
    int y0 = floor(y), y1 = y0 + 1;

    if(x0 < 0 || y0 < 0) return glm::dvec3(-1, -1, -1);
    if(x1 >= img.width() || y1 >= img.height()) return glm::dvec3(-1, -1, -1);

    double c0 = x - x0, c0c = 1 - c0;
    double c1 = y - y0, c1c = 1 - c1;

    QRgb p00 = img.pixel(x0, y0);
    QRgb p01 = img.pixel(x1, y0);
    QRgb p10 = img.pixel(x0, y1);
    QRgb p11 = img.pixel(x1, y1);

    double r = c0c * c1c * qRed(p00) + c0c * c1 * qRed(p01) + c0 * c1c * qRed(p10) + c0 * c1 * qRed(p11);
    double g = c0c * c1c * qGreen(p00) + c0c * c1 * qGreen(p01) + c0 * c1c * qGreen(p10) + c0 * c1 * qGreen(p11);
    double b = c0c * c1c * qBlue(p00) + c0c * c1 * qBlue(p01) + c0 * c1c * qBlue(p10) + c0 * c1 * qBlue(p11);

    return glm::dvec3(r, g, b);
  }

  inline pair<set<int>, vector<int>> FindTrianglesIndices(const QImage& img) {
    int w = img.width(), h = img.height();
    set<int> S;
    vector<int> indices_map(w*h);
    for(int i=0, pidx = 0;i<h;++i) {
      for(int j=0;j<w;++j, ++pidx) {
        QRgb pix = img.pixel(j, i);
        unsigned char r = static_cast<unsigned char>(qRed(pix));
        unsigned char g = static_cast<unsigned char>(qGreen(pix));
        unsigned char b = static_cast<unsigned char>(qBlue(pix));

        if(r == 0 && g == 0 && b == 0) {
          indices_map[pidx] = -1;
          continue;
        }
        else {
          int idx;
          decode_index(r, g, b, idx);
          S.insert(idx);
          indices_map[pidx] = idx;
        }
      }
    }
    return make_pair(S, indices_map);
  }
  static QImage TransferColor(const QImage& source, const QImage& target,
                              const vector<int>& valid_pixels_s,
                              const vector<int>& valid_pixels_t) {
    // Make a copy
    QImage result = source;

    const int num_rows_s = source.height(), num_cols_s = source.width();
    const int num_rows_t = target.height(), num_cols_t = target.width();
    const size_t num_pixels_s = valid_pixels_s.size();
    const size_t num_pixels_t = valid_pixels_t.size();

    Matrix3d RGB2LMS, LMS2RGB;
    RGB2LMS << 0.3811, 0.5783, 0.0402,
               0.1967, 0.7244, 0.0782,
               0.0241, 0.1288, 0.8444;
    LMS2RGB << 4.4679, -3.5873, 0.1193,
              -1.2186, 2.3809, -0.1624,
               0.0497, -0.2439, 1.2045;

    Matrix3d b, c, b2, c2;
    b << 1.0/sqrt(3.0), 0, 0,
         0, 1.0/sqrt(6.0), 0,
         0, 0, 1.0/sqrt(2.0);
    c << 1, 1, 1,
         1, 1, -2,
         1, -1, 0;
    b2 << sqrt(3.0)/3.0, 0, 0,
          0, sqrt(6.0)/6.0, 0,
          0, 0, sqrt(2.0)/2.0;
    c2 << 1, 1, 1,
          1, 1, -1,
          1, -2, 0;
    Matrix3d LMS2lab = b * c;
    Matrix3d lab2LMS = c2 * b2;

    auto unpack_pixel = [](QRgb pix) {
      int r = max(1, qRed(pix)), g = max(1, qGreen(pix)), b = max(1, qBlue(pix));
      return make_tuple(r, g, b);
    };

    auto compute_image_stats = [&](const QImage& img, const vector<int>& valid_pixels) {
      const size_t num_pixels = valid_pixels.size();
      const int num_cols = img.width(), num_rows  = img.height();
      MatrixXd pixels(3, num_pixels);

      cout << num_cols << 'x' << num_rows << endl;

      for(size_t i=0;i<num_pixels;++i) {
        int y = valid_pixels[i] / num_cols;
        int x = valid_pixels[i] % num_cols;

        int r, g, b;
        tie(r, g, b) = unpack_pixel(img.pixel(x, y));
        pixels.col(i) = Vector3d(r / 255.0, g / 255.0, b / 255.0);
      }

      MatrixXd pixels_LMS = RGB2LMS * pixels;

      for(int i=0;i<3;i++) {
        for(int j=0;j<num_pixels;++j) {
          pixels_LMS(i, j) = log10(pixels_LMS(i, j));
        }
      }

      MatrixXd pixels_lab = LMS2lab * pixels_LMS;

      Vector3d mean = pixels_lab.rowwise().mean();
      Vector3d stdev(0, 0, 0);
      for(int i=0;i<num_pixels;++i) {
        Vector3d diff = pixels_lab.col(i) - mean;
        stdev += Vector3d(diff[0]*diff[0], diff[1]*diff[1], diff[2]*diff[2]);
      }
      stdev /= (num_pixels - 1);

      for(int i=0;i<3;++i) stdev[i] = sqrt(stdev[i]);

      cout << "mean: " << mean << endl;
      cout << "std: " << stdev << endl;

      return make_tuple(pixels_lab, mean, stdev);
    };

    // Compute stats of both images
    MatrixXd lab_s, lab_t;
    Vector3d mean_s, std_s, mean_t, std_t;
    tie(lab_s, mean_s, std_s) = compute_image_stats(source, valid_pixels_s);
    tie(lab_t, mean_t, std_t) = compute_image_stats(target, valid_pixels_t);

    // Do the transfer
    MatrixXd res(3, num_pixels_s);
    for(int i=0;i<3;++i) {
      for(int j=0;j<num_pixels_s;++j) {
        //res(i, j) = (lab_s(i, j) - mean_s[i]) * std_t[i] / std_s[i] + mean_t[i];
        res(i, j) = (lab_s(i, j) - mean_s[i]) + mean_t[i];
      }
    }

    MatrixXd LMS_res = lab2LMS * res;
    for(int i=0;i<3;++i) {
      for(int j=0;j<num_pixels_s;++j) {
        LMS_res(i, j) = pow(10, LMS_res(i, j));
      }
    }

    MatrixXd est_im = LMS2RGB * LMS_res;
    for(size_t i=0;i<num_pixels_s;++i) {
      int y = valid_pixels_s[i] / num_cols_s;
      int x = valid_pixels_s[i] % num_cols_s;
      result.setPixel(x, y, qRgb(clamp<double>(est_im(0, i) * 255.0, 0., 255.),
                                 clamp<double>(est_im(1, i) * 255.0, 0., 255.),
                                 clamp<double>(est_im(2, i) * 255.0, 0., 255.)));
    }
    return result;
  }
}

template <typename Constraint>
class MultiImageReconstructor {
public:
  MultiImageReconstructor():
    enable_selection(true),
    enable_failure_detection(true),
    direct_multi_recon(false) {}

  void LoadModel(const string& filename) {
    model = MultilinearModel(filename);
    single_recon.LoadModel(filename);
  }
  void LoadPriors(const string& filename_id, const string& filename_exp) {
    prior.load(filename_id, filename_exp);
    single_recon.LoadPriors(filename_id, filename_exp);
  }
  void SetContourIndices(const vector<vector<int>>& contour_indices_in) {
    contour_indices = contour_indices_in;
    single_recon.SetContourIndices(contour_indices_in);
  }
  void SetMesh(const BasicMesh& mesh) {
    template_mesh = mesh;
  }
  void SetIndices(const vector<int>& indices) {
    init_indices = indices;
  }

  void AddImagePointsPair(const string& filename, const pair<QImage, vector<Constraint>>& p) {
    image_filenames.push_back(filename);
    image_points_pairs.push_back(p);
  }

  bool Reconstruct();

  const Vector3d& GetRotation(int imgidx) const { return param_sets[imgidx].model.R; }
  const Vector3d& GetTranslation(int imgidx) const { return param_sets[imgidx].model.T; }
  const VectorXd& GetIdentityWeights(int imgidx) const { return param_sets[imgidx].model.Wid; }
  const VectorXd& GetExpressionWeights(int imgidx) const { return param_sets[imgidx].model.Wexp_FACS; }
  const Tensor1& GetGeometry(int imgidx) const {
    model.ApplyWeights(GetIdentityWeights(imgidx), GetExpressionWeights(imgidx));
    return model.GetTM();
  }
  const CameraParameters GetCameraParameters(int imgidx) const { return param_sets[imgidx].cam; }
  const vector<int> GetIndices(int imgidx) const { return param_sets[imgidx].indices; }
  vector<int> GetUpdatedIndices(int imgidx) const {
    vector<int> idxs;
    for(int i=0;i<param_sets[imgidx].recon.cons.size();++i) {
      idxs.push_back(param_sets[imgidx].recon.cons[i].vidx);
    }
    return idxs;
  }

  void SetSelectionState(bool val) { enable_selection = val; }
  void SetFailureDetectionState(bool val) { enable_failure_detection = val; }
  void SetDirectMultiRecon(bool val) { direct_multi_recon = val; }
  void SetProgressiveReconState(bool val) { enable_progressive_recon = val; }

protected:
  void VisualizeReconstructionResult(const fs::path& folder, int i, bool scale_output=true) {
    // Visualize the reconstruction results
    #if 0
    MeshVisualizer w("reconstruction result", param_sets[i].mesh);
    w.BindConstraints(image_points_pairs[i].second);
    w.BindImage(image_points_pairs[i].first);
    w.BindLandmarks(init_indices);

    w.BindUpdatedLandmarks(param_sets[i].indices);
    w.SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
    w.SetCameraParameters(param_sets[i].cam);
    w.resize(image_points_pairs[i].first.width(), image_points_pairs[i].first.height());
    w.show();
    w.paintGL();
    w.update();

    QImage recon_image = w.grabFrameBuffer();
    fs::path image_path = fs::path(image_filenames[i]);

    recon_image.save( (folder / fs::path(image_path.stem().string() + ".png")).string().c_str() );
    #else
    int imgw = image_points_pairs[i].first.width();
    int imgh = image_points_pairs[i].first.height();
    if(scale_output) {
      const int target_size = 640;
      double scale = static_cast<double>(target_size) / imgw;
      imgw *= scale;
      imgh *= scale;
    }

    const string home_directory = QDir::homePath().toStdString();
    cout << "Home dir: " << home_directory << endl;

    OffscreenMeshVisualizer visualizer(imgw, imgh);

    // Always compute normal
    param_sets[i].mesh.ComputeNormals();

    visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
    visualizer.SetRenderMode(OffscreenMeshVisualizer::MeshAndImage);
    visualizer.BindMesh(param_sets[i].mesh);
    visualizer.BindImage(image_points_pairs[i].first);
    visualizer.SetCameraParameters(param_sets[i].cam);
    visualizer.SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
    visualizer.SetIndexEncoded(false);
    visualizer.SetEnableLighting(true);
    visualizer.LoadRenderingSettings(home_directory + "/Data/Settings/blendshape_vis_ao.json");

    QImage img = visualizer.Render(true);
    fs::path image_path = fs::path(image_filenames[i]);
    img.save((folder / fs::path(image_path.stem().string() + ".png")).string().c_str());
    #endif
  }

private:
  MultilinearModel model;
  MultilinearModelPrior prior;
  vector<vector<int>> contour_indices;
  vector<int> init_indices;
  BasicMesh template_mesh;

  struct ParameterSet {
    vector<int> indices;
    BasicMesh mesh;

    CameraParameters cam;
    ModelParameters model;
    ReconstructionParameters<Constraint> recon;
    OptimizationParameters opt;
    ReconstructionStats stats;
    string img_filename;
  };

  // Input image points pairs
  vector<pair<QImage, vector<Constraint>>> image_points_pairs;
  vector<string> image_filenames;

  // AAM model for consistent set selection
  aam::AAMModel aam;

  // A set of parameters for each image
  vector<ParameterSet> param_sets;

  // The worker for single image reconstruction
  SingleImageReconstructor<Constraint> single_recon;

  bool enable_selection;
  bool enable_failure_detection;
  bool enable_progressive_recon;
  bool direct_multi_recon;
};

namespace {
  void safe_create(const fs::path& p) {
    if(fs::exists(p)) fs::remove_all(p);
    fs::create_directory(p);
  }
}

template <typename Constraint>
bool MultiImageReconstructor<Constraint>::Reconstruct() {
  cout << "Reconstruction begins..." << endl;

  const string home_directory = QDir::homePath().toStdString();
  cout << "Home dir: " << home_directory << endl;

  // Preparing necessary stuff
  const int tex_size = 2048;
  const string albedo_index_map_filename(home_directory + "/Data/Multilinear/albedo_index.png");
  const string albedo_pixel_map_filename(home_directory + "/Data/Multilinear/albedo_pixel.png");
  const string valid_faces_indices_filename(home_directory + "/Data/Multilinear/face_region_indices.txt");

  QImage albedo_index_map;
  // Get the albedo index map
  if(QFile::exists(albedo_index_map_filename.c_str())) {
    message("loading index map for albedo.");
    albedo_index_map = QImage(albedo_index_map_filename.c_str());
    albedo_index_map.save("albedo_index.png");
  } else {
    cerr << "albedo index map does not exist. Abort." << endl;
    exit(1);
  }

  auto valid_faces_indices_quad = LoadIndices(valid_faces_indices_filename);
  // @HACK each quad face is triangulated, so the indices change from i to [2*i, 2*i+1]
  vector<int> valid_faces_indices;
  for(auto fidx : valid_faces_indices_quad) {
    valid_faces_indices.push_back(fidx*2);
    valid_faces_indices.push_back(fidx*2+1);
  }

  // Compute the barycentric coordinates for each pixel
  vector<vector<PixelInfo>> albedo_pixel_map(tex_size, vector<PixelInfo>(tex_size));

  // Generate pixel map for albedo
  bool gen_pixel_map = false;
  QImage pixel_map_image;
  if(QFile::exists(albedo_pixel_map_filename.c_str())) {
    pixel_map_image = QImage(albedo_pixel_map_filename.c_str());

    message("generating pixel map for albedo ...");
    boost::timer::auto_cpu_timer t("pixel map for albedo generation time = %w seconds.\n");

    for(int i=0;i<tex_size;++i) {
      for(int j=0;j<tex_size;++j) {
        QRgb pix = albedo_index_map.pixel(j, i);
        unsigned char r = static_cast<unsigned char>(qRed(pix));
        unsigned char g = static_cast<unsigned char>(qGreen(pix));
        unsigned char b = static_cast<unsigned char>(qBlue(pix));
        if(r == 0 && g == 0 && b == 0) continue;
        int fidx;
        decode_index(r, g, b, fidx);

        QRgb bcoords_pix = pixel_map_image.pixel(j, i);

        float x = static_cast<float>(qRed(bcoords_pix)) / 255.0f;
        float y = static_cast<float>(qGreen(bcoords_pix)) / 255.0f;
        float z = static_cast<float>(qBlue(bcoords_pix)) / 255.0f;
        albedo_pixel_map[i][j] = PixelInfo(fidx, glm::vec3(x, y, z));
      }
    }
    message("done.");
  } else {
    cerr << "albedo pixel map does not exist. Abort." << endl;
    exit(1);
  }

  vector<vector<glm::dvec3>> mean_texture(tex_size, vector<glm::dvec3>(tex_size, glm::dvec3(0, 0, 0)));
  cv::Mat mean_texture_mat(tex_size, tex_size, CV_64FC3);
  vector<vector<double>> mean_texture_weight(tex_size, vector<double>(tex_size, 0));
  QImage mean_texture_image;

  // Misc stuff
  cout << image_filenames.size() << endl;
  fs::path image_path = fs::path(image_filenames.front()).parent_path();
  fs::path result_path = image_path / fs::path("multi_recon");
  cout << "creating directory " << result_path.string() << endl;
  safe_create(result_path);
  cout << "directory created ..." << endl;

  // Initialize the parameter sets
  param_sets.resize(image_points_pairs.size());
  for(size_t i=0;i<param_sets.size();++i) {
    auto& params = param_sets[i];
    params.img_filename = fs::path(image_filenames[i]).filename().string();
    params.indices = init_indices;
    params.mesh = template_mesh;

    const int image_width = image_points_pairs[i].first.width();
    const int image_height = image_points_pairs[i].first.height();

    // camera parameters
    cout << image_width << "x" << image_height << endl;
    params.cam = CameraParameters::DefaultParameters(image_width, image_height);
    cout << params.cam.image_size.x << ", " << params.cam.image_size.y << endl;

    // model parameters
    params.model = ModelParameters::DefaultParameters(prior.Uid, prior.Uexp);

    // reconstruction parameters
    params.recon.cons = image_points_pairs[i].second;
    params.recon.imageWidth = image_width;
    params.recon.imageHeight = image_height;
  }

  const int num_images = image_points_pairs.size();

  // Initialize AAM model
  auto constraints_to_mat = [=](const vector<Constraint>& constraints, int h) {
    const int npoints = constraints.size();
    cv::Mat m(npoints, 2, CV_64FC1);
    for(int j=0;j<npoints;++j) {
      m.at<double>(j, 0) = constraints[j].data.x;
      m.at<double>(j, 1) = h - constraints[j].data.y;
    }
    return m;
  };

  vector<int> inliers;
  if(enable_failure_detection) {
    vector<QImage> images(image_points_pairs.size());
    vector<cv::Mat> points(image_points_pairs.size());

    // Collect input images and points
    for(int i=0;i<image_points_pairs.size();++i) {
      images[i] = image_points_pairs[i].first;
      points[i] = constraints_to_mat(image_points_pairs[i].second,
                                     image_points_pairs[i].first.height());
    }

    aam.SetOutputPath(result_path.string());
    aam.SetImages(images);
    aam.SetPoints(points);
    aam.Preprocess();
    aam.SetErrorMetric(aam::AAMModel::Hybrid);

    // For Debugging
    inliers = aam.FindInliers_Iterative();
  } else {
    inliers.resize(num_images);
    iota(inliers.begin(), inliers.end(), 0);
  }

  VectorXd identity_centroid;

  // Main reconstruction loop
  //  1. Use single image reconstructor to do per-image reconstruction first
  //  2. Select a consistent set of images for joint reconstruction
  //  3. Convergence test. If not converged, goto step 1.
  const int max_iters_main_loop = enable_progressive_recon?3:1;
  int iters_main_loop = 0;

  vector<MatrixXd> identity_weights_history;
  vector<VectorXd> identity_weights_centroid_history;

  vector<int> consistent_set, final_chosen_set;
  // Initialize the consistent set to inliers
#if 0
  consistent_set.resize(num_images);
  iota(consistent_set.begin(), consistent_set.end(), 0);
#else
  consistent_set = inliers;
#endif

  while(iters_main_loop++ < max_iters_main_loop){
    fs::path step_result_path = result_path / fs::path("step" + to_string(iters_main_loop));
    safe_create(step_result_path);

    // Single image reconstruction step
    OptimizationParameters opt_params = OptimizationParameters::Defaults();
    opt_params.w_prior_id = 10 * pow(iters_main_loop, 0.25);
    opt_params.w_prior_exp = 10;
    opt_params.num_initializations = 1;
    opt_params.perturbation_range = 0.01;
    opt_params.errorThreshold = 0.01;

    fs::path step_single_recon_result_path = step_result_path / fs::path("single_recon");
    safe_create(step_single_recon_result_path);
    for(int i=0;i<num_images;++i) {
      single_recon.SetMesh(param_sets[i].mesh);
      single_recon.SetIndices(param_sets[i].indices);
      single_recon.SetImageSize(param_sets[i].recon.imageWidth, param_sets[i].recon.imageHeight);
      single_recon.SetConstraints(param_sets[i].recon.cons);

      single_recon.SetInitialParameters(param_sets[i].model, param_sets[i].cam);
      if(iters_main_loop > 1) single_recon.SetIdentityPrior(identity_centroid);

      // Perform reconstruction
      if(!direct_multi_recon) {
        boost::timer::auto_cpu_timer t("Single image reconstruction finished in %w seconds.\n");
        single_recon.Reconstruct(opt_params);
      } else continue;

      // Store results
      auto tm = single_recon.GetGeometry();
      param_sets[i].mesh.UpdateVertices(tm);
      param_sets[i].mesh.ComputeNormals();
      param_sets[i].model = single_recon.GetModelParameters();
      param_sets[i].indices = single_recon.GetIndices();
      param_sets[i].cam = single_recon.GetCameraParameters();

      if (true) {
        VisualizeReconstructionResult(step_single_recon_result_path, i);
        fs::path image_path = fs::path(image_filenames[i]);
        single_recon.SaveReconstructionResults( (step_single_recon_result_path / fs::path(image_path.stem().string() + ".res")).string());
      }
    }

    // TODO Parameters estimation step, choose a consistent set of images for joint
    // optimization
    MatrixXd identity_weights(param_sets[0].model.Wid.rows(), num_images);
    for(int i=0;i<num_images;++i) {
      identity_weights.col(i) = param_sets[i].model.Wid;
    }

    identity_weights_history.push_back(identity_weights);

    // Remove outliers
    fs::path selection_result_path = step_result_path / fs::path("selection");
    safe_create(selection_result_path);

    int selection_method = enable_selection?1:2;

    switch(selection_method) {
      case 0: {
        const double ratios[] = {0.0, 0.4, 0.6, 0.8};
        consistent_set = StatsUtils::FindConsistentSet(identity_weights, 0.5, ratios[iters_main_loop] * num_images, &identity_centroid);
        assert(consistent_set.size() > 0);
        for(auto i : consistent_set) {
          VisualizeReconstructionResult(selection_result_path, i);
        }
        break;
      }
      case 1: {
        double ratios[] = {0.0, 0.4, 0.6, 0.8};

        // HACK for testing the system without progressive reconstruction
        if(max_iters_main_loop == 1) ratios[1] = 0.8;

        // Take the first few as good shape
        int k = max(1, static_cast<int>(ratios[iters_main_loop] * num_images));

        consistent_set.clear();

        auto take_first_k = [](vector<pair<int, double>> stats, int k) {
          set<int> subset;
          std::sort(stats.begin(), stats.end(), [](pair<int,double> a, pair<int, double> b){
            return a.second < b.second;
          });
          for(int i=0;i<k;++i) {
            subset.insert(stats[i].first);
          }
          return subset;
        };

        // Choose the ones with smallest error, not very useful
        vector<pair<int, double>> errors(num_images);
        for(int i=0;i<num_images;++i) {
          errors[i] = make_pair(i, param_sets[i].stats.avg_error);
        }
        auto subset_error = take_first_k(errors, num_images);
        for(auto sx : subset_error) cout << sx << ' '; cout << endl;

        // Compute the distance to mean identity weights, choose the close ones
        VectorXd mean_identity = StatsUtils::mean(identity_weights, 2);
        vector<pair<int, double>> d_identity(num_images);
        for(int i=0;i<num_images;++i) {
          d_identity[i] = make_pair(i, (identity_weights.col(i) - mean_identity).norm());
        }
        auto subset_identity = take_first_k(d_identity, k);
        for(auto sx : subset_identity) cout << sx << ' '; cout << endl;

        // Compute the norm of the expression weights, choose the smaller ones
        vector<pair<int, double>> n_expression(num_images);
        for(int i=0;i<num_images;++i) {
          n_expression[i] = make_pair(i, (param_sets[i].model.Wexp_FACS).norm());
        }

        auto subset_expression = take_first_k(n_expression, 0.8 * num_images);
        for(auto sx : subset_expression) cout << sx << ' '; cout << endl;

        #if 1
        if(iters_main_loop == 1) {
          // Compute the RMSE of color transferred texture
          // Collect texture information from each input (image, mesh) pair to obtain mean texture
          bool generate_mean_texture = true;

          vector<vector<int>> face_indices_maps;
          {
            for(int img_i=0;img_i<num_images;++img_i) {
              const auto& mesh = param_sets[img_i].mesh;

              // for each image bundle, render the mesh to FBO with culling to get the visible triangles
              OffscreenMeshVisualizer visualizer(image_points_pairs[img_i].first.width(),
                                                 image_points_pairs[img_i].first.height());
              visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
              visualizer.SetRenderMode(OffscreenMeshVisualizer::Mesh);
              visualizer.BindMesh(param_sets[img_i].mesh);
              visualizer.SetCameraParameters(param_sets[img_i].cam);
              visualizer.SetMeshRotationTranslation(param_sets[img_i].model.R, param_sets[img_i].model.T);
              visualizer.SetIndexEncoded(true);
              visualizer.SetEnableLighting(false);
              QImage img = visualizer.Render();
              //img.save("mesh.png");

              // find the visible triangles from the index map
              auto triangles_indices_pair = FindTrianglesIndices(img);
              set<int> triangles = triangles_indices_pair.first;
              face_indices_maps.push_back(triangles_indices_pair.second);
              cerr << "triangles = " << triangles.size() << endl;

              // get the projection parameters
              glm::dmat4 Rmat = glm::eulerAngleYXZ(param_sets[img_i].model.R[0],
                                                   param_sets[img_i].model.R[1],
                                                   param_sets[img_i].model.R[2]);
              glm::dmat4 Tmat = glm::translate(glm::dmat4(1.0),
                                               glm::dvec3(param_sets[img_i].model.T[0],
                                                          param_sets[img_i].model.T[1],
                                                          param_sets[img_i].model.T[2]));
              glm::dmat4 Mview = Tmat * Rmat;

              // FOR DEBUGGING
              #if 0
              // for each visible triangle, compute the coordinates of its 3 corners
              QImage img_vertices = img;
              vector<vector<glm::dvec3>> triangles_projected;
              for(auto tidx : triangles) {
                auto face_i = mesh.face(tidx);
                auto v0_mesh = mesh.vertex(face_i[0]);
                auto v1_mesh = mesh.vertex(face_i[1]);
                auto v2_mesh = mesh.vertex(face_i[2]);
                glm::dvec3 v0_tri = ProjectPoint(glm::dvec3(v0_mesh[0], v0_mesh[1], v0_mesh[2]), Mview, param_sets[img_i].cam);
                glm::dvec3 v1_tri = ProjectPoint(glm::dvec3(v1_mesh[0], v1_mesh[1], v1_mesh[2]), Mview, param_sets[img_i].cam);
                glm::dvec3 v2_tri = ProjectPoint(glm::dvec3(v2_mesh[0], v2_mesh[1], v2_mesh[2]), Mview, param_sets[img_i].cam);
                triangles_projected.push_back(vector<glm::dvec3>{v0_tri, v1_tri, v2_tri});


                img_vertices.setPixel(v0_tri.x, img.height()-1-v0_tri.y, qRgb(255, 255, 255));
                img_vertices.setPixel(v1_tri.x, img.height()-1-v1_tri.y, qRgb(255, 255, 255));
                img_vertices.setPixel(v2_tri.x, img.height()-1-v2_tri.y, qRgb(255, 255, 255));
              }
              img_vertices.save("mesh_with_vertices.png");
              #endif

              #define DEBUG_RECON 1 // for visualizing large scale recon selection related data

              message("generating mean texture...");
              message("collecting texels...");
              if(generate_mean_texture) {
                // for each pixel in the texture map, use backward projection to obtain pixel value in the input image
                // accumulate the texels in average texel map
                for(int ti=0;ti<tex_size;++ti) {
                  for(int tj=0;tj<tex_size;++tj) {
                    PixelInfo pix_ij = albedo_pixel_map[ti][tj];

                    // skip if the triangle is not visible
                    if(triangles.find(pix_ij.fidx) == triangles.end()) continue;

                    auto face_i = mesh.face(pix_ij.fidx);

                    auto v0_mesh = mesh.vertex(face_i[0]);
                    auto v1_mesh = mesh.vertex(face_i[1]);
                    auto v2_mesh = mesh.vertex(face_i[2]);

                    auto v = v0_mesh * pix_ij.bcoords.x + v1_mesh * pix_ij.bcoords.y + v2_mesh * pix_ij.bcoords.z;

                    glm::dvec3 v_img = ProjectPoint(glm::dvec3(v[0], v[1], v[2]), Mview, param_sets[img_i].cam);

                    // take the pixel from the input image through bilinear sampling
                    glm::dvec3 texel = bilinear_sample(image_points_pairs[img_i].first, v_img.x, image_points_pairs[img_i].first.height()-1-v_img.y);

                    if(texel.r < 0 && texel.g < 0 && texel.b < 0) continue;

                    mean_texture[ti][tj] += texel;
                    mean_texture_weight[ti][tj] += 1.0;
                  }
                }
              }
            }
            message("done.");

            try {
              // [Optional]: render the mesh with texture to verify the texel values
              if(generate_mean_texture) {
                message("computing mean texture...");
                mean_texture_image = QImage(tex_size, tex_size, QImage::Format_ARGB32);
                mean_texture_image.fill(0);
                for(int ti=0; ti<tex_size; ++ti) {
                  for (int tj=0; tj<(tex_size/2); ++tj) {
                    double weight_ij = mean_texture_weight[ti][tj];
                    double weight_ij_s = mean_texture_weight[ti][tex_size-1-tj];

                    if(weight_ij == 0 && weight_ij_s == 0) {
                      mean_texture_mat.at<cv::Vec3d>(ti, tj) = cv::Vec3d(0, 0, 0);
                      continue;
                    } else {
                      glm::dvec3 texel = (mean_texture[ti][tj] + mean_texture[ti][tex_size-1-tj]) / (weight_ij + weight_ij_s);
                      mean_texture[ti][tj] = texel;
                      mean_texture[ti][tex_size-1-tj] = texel;
                      mean_texture_image.setPixel(tj, ti, qRgb(texel.r, texel.g, texel.b));
                      mean_texture_image.setPixel(tex_size-1-tj, ti, qRgb(texel.r, texel.g, texel.b));

                      mean_texture_mat.at<cv::Vec3d>(ti, tj) = cv::Vec3d(texel.x, texel.y, texel.z);
                      mean_texture_mat.at<cv::Vec3d>(ti, tex_size-1-tj) = cv::Vec3d(texel.x, texel.y, texel.z);
                    }
                  }
                }
                message("done.");

                cv::resize(mean_texture_mat, mean_texture_mat, cv::Size(), 0.25, 0.25);
                //cv::Mat mean_texture_refined_mat = mean_texture_mat.clone();
                cv::Mat mean_texture_refined_mat;
                {
                  boost::timer::auto_cpu_timer timer_solve(
                    "[Joint optimization] Mean texture generation = %w seconds.\n");
                  #if 1
                  cv::GaussianBlur(mean_texture_mat, mean_texture_refined_mat, cv::Size(5, 5), 3.0);
                  mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 5.0, 30.0, 0.5);
                  mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
                  /*
                  mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 10.0, 30.0, 0.5);
                  mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
                  mean_texture_refined_mat = StatsUtils::MeanShiftSegmentation(mean_texture_refined_mat, 20.0, 30.0, 0.5);
                  mean_texture_refined_mat = 0.25 * mean_texture_mat + 0.75 * mean_texture_refined_mat;
                  */
                  cv::resize(mean_texture_refined_mat, mean_texture_refined_mat, cv::Size(), 4.0, 4.0);
                  #else
                  cv::Mat mean_texture_refined_mat = mean_texture_mat;
                  #endif
                }

                QImage mean_texture_image_refined(tex_size, tex_size, QImage::Format_ARGB32);
                for(int ti=0;ti<tex_size;++ti) {
                  for(int tj=0;tj<tex_size;++tj) {
                    cv::Vec3d pix = mean_texture_refined_mat.at<cv::Vec3d>(ti, tj);
                    mean_texture_image_refined.setPixel(tj, ti, qRgb(pix[0], pix[1], pix[2]));
                  }
                }

                #if DEBUG_RECON
                mean_texture_image.save( (step_result_path / fs::path("mean_texture.png")).string().c_str() );
                mean_texture_image_refined.save( (step_result_path / fs::path("mean_texture_refined.png")).string().c_str() );
                #endif
                mean_texture_image = mean_texture_image_refined;
              }
            } catch(exception& e) {
              cerr << e.what() << endl;
              exit(1);
            }
          }
        }

        vector<pair<int, double>> d_texture(num_images);

        // Rendering the albedo to each image
        vector<QImage> albedo_images(num_images);
        //#pragma omp parallel for
        for(int i=0;i<num_images;++i) {
          // for each image bundle, render the mesh to FBO with culling to get the visible triangles
          OffscreenMeshVisualizer visualizer(image_points_pairs[i].first.width(),
                                             image_points_pairs[i].first.height());
          visualizer.SetMVPMode(OffscreenMeshVisualizer::CamPerspective);
          visualizer.SetRenderMode(OffscreenMeshVisualizer::TexturedMesh);
          visualizer.BindMesh(param_sets[i].mesh);
          visualizer.BindTexture(mean_texture_image);
          visualizer.SetCameraParameters(param_sets[i].cam);
          visualizer.SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
          visualizer.SetFacesToRender(valid_faces_indices);

          vector<float> depth_i;
          tie(albedo_images[i],depth_i) = visualizer.RenderWithDepth();

          auto unpack_pixel = [](QRgb pix) {
            return Vector3d(qRed(pix)/255.0, qGreen(pix)/255.0, qBlue(pix)/255.0);
          };
          int img_w = image_points_pairs[i].first.width();
          int img_h = image_points_pairs[i].first.height();

          vector<int> valid_pixels_map_i;
          for(int y=0;y<img_h;++y) {
            for(int x=0;x<img_w;++x) {
              float dval = depth_i[(img_h-1-y)*img_w+x];
              if(dval<1) {
                valid_pixels_map_i.push_back(y*img_w + x);
                QRgb pix1 = albedo_images[i].pixel(x, y);
                albedo_images[i].setPixel(x, y, qRgb(qBlue(pix1), qGreen(pix1), qRed(pix1)));
              }
            }
          }

          albedo_images[i] = TransferColor(albedo_images[i], image_points_pairs[i].first,
                                           valid_pixels_map_i, valid_pixels_map_i);
          #if DEBUG_RECON
          albedo_images[i].save( (step_result_path / fs::path("albedo_" + std::to_string(i) + ".png")).string().c_str() );
          #endif

          // compute texture difference

          double diff_i = 0;
          int valid_count = 0;
          #if DEBUG_RECON
          QImage depth_image = albedo_images[i];
          depth_image.fill(0);
          #endif
          for(int y=0;y<img_h;++y) {
            for(int x=0;x<img_w;++x) {
              float dval = depth_i[(img_h-1-y)*img_w+x];
              if(dval<1) {
                #if DEBUG_RECON
                depth_image.setPixel(x, y, qRgb(dval*255, 0, (1-dval)*255));
                #endif
                valid_count++;
                QRgb pix1 = albedo_images[i].pixel(x, y);
                QRgb pix2 = image_points_pairs[i].first.pixel(x, y);
                auto p1 = unpack_pixel(pix1);
                auto p2 = unpack_pixel(pix2);
                double dr = p1[0] - p2[0];
                double dg = p1[1] - p2[1];
                double db = p1[2] - p2[2];
                diff_i += dr*dr+dg*dg+db*db;
              }
            }
          }
          d_texture[i] = make_pair(i, diff_i/valid_count);
          #if DEBUG_RECON
          depth_image.save( (step_result_path / fs::path("depth_" + std::to_string(i) + ".png")).string().c_str() );
          #endif
        }
        auto subset_texture = take_first_k(d_texture, k);
        for(auto sx : subset_texture) cout << sx << ' '; cout << endl;

        #endif

        // Merge them into a consistent set
        set<int> final_set(subset_identity.begin(), subset_identity.end());

        for(int i=0;i<num_images;++i) {
          if(subset_identity.count(i)) {

#if 1
            // Use expression as a condition
            bool exclude = (subset_expression.count(i) == 0) || (subset_error.count(i) == 0) ||
                           (subset_texture.count(i) == 0) || (find(inliers.begin(), inliers.end(), i) == inliers.end());

#else
            // Use only recon error and texture metric
            bool exclude = (subset_error.count(i) == 0) || (subset_texture.count(i) == 0);
#endif

            if(exclude) final_set.erase(i);
          }
        }

        // rare case, we go with the mean identity
        if(final_set.empty()) {
          final_set = take_first_k(d_identity, 1);
        }

        consistent_set.assign(final_set.begin(), final_set.end());

        for(auto i : consistent_set) {
          VisualizeReconstructionResult(selection_result_path, i);
        }
        break;
      }
      case 2: {
        // nothing to do, just use whatever consistent_set is
        break;
      }
    }

    // Compute the centroid of the consistent set
    identity_centroid = VectorXd::Zero(param_sets[0].model.Wid.rows());
    for(auto i : consistent_set) {
      cout << i << endl;
      identity_centroid += param_sets[i].model.Wid;
    }
    identity_centroid /= consistent_set.size();

    // Update the identity weights for all images
    for(auto& param : param_sets) {
      param.model.Wid = identity_centroid;
    }

    // Joint reconstruction step, obtain refined identity weights
    int num_iters_joint_optimization = (iters_main_loop == max_iters_main_loop)?4:3;

    // Just one-pass optimization
    opt_params.num_initializations = 1;

    for(int iters_joint_optimization=0;
        iters_joint_optimization<num_iters_joint_optimization;
        ++iters_joint_optimization){
      // [Joint reconstruction] step 1: estimate pose and expression weights individually

      // In the final iteration, no need to refine the identity weights anymore
      if((iters_joint_optimization == num_iters_joint_optimization - 1) && (iters_main_loop == max_iters_main_loop)) {
        // Store the final selection
        // HACK try to use the inliners as final_chosen_set to produce more point clouds
        #if 1
        final_chosen_set = consistent_set;
        #else
        // No good!
        final_chosen_set = inliers;
        #endif

        // Reset consistent_set so all images will be reconstructed in this iteration
        consistent_set.resize(num_images);
        for(int i=0;i<num_images;++i) consistent_set[i] = i;
      }

      fs::path joint_pre_result_path = step_result_path / fs::path("joint_recon_" + to_string(iters_joint_optimization) + "_pre");
      safe_create(joint_pre_result_path);

      for(auto i : consistent_set) {
        single_recon.SetMesh(param_sets[i].mesh);
        single_recon.SetIndices(param_sets[i].indices);
        single_recon.SetImageSize(param_sets[i].recon.imageWidth, param_sets[i].recon.imageHeight);
        single_recon.SetConstraints(param_sets[i].recon.cons);

        single_recon.SetInitialParameters(param_sets[i].model, param_sets[i].cam);
        single_recon.SetOptimizationMode(
          static_cast<typename SingleImageReconstructor<Constraint>::OptimizationMode>(
            SingleImageReconstructor<Constraint>::Pose
            | SingleImageReconstructor<Constraint>::Expression
            | SingleImageReconstructor<Constraint>::FocalLength));
        {
          boost::timer::auto_cpu_timer t("Single image reconstruction finished in %w seconds.\n");
          single_recon.Reconstruct(opt_params);
        }

        // Store results
        auto tm = single_recon.GetGeometry();
        param_sets[i].mesh.UpdateVertices(tm);
        param_sets[i].model = single_recon.GetModelParameters();
        param_sets[i].indices = single_recon.GetIndices();
        param_sets[i].cam = single_recon.GetCameraParameters();

        if (true) {
          // Visualize the reconstruction results
          VisualizeReconstructionResult(joint_pre_result_path, i);
        }
      }

      if((iters_joint_optimization == num_iters_joint_optimization - 1) && (iters_main_loop == max_iters_main_loop)) {
        // In the final iteration, no need to refine the identity weights anymore
        break;
      }

      // [Joint reconstruction] step 2: estimate identity weights jointly
      {
        fs::path joint_post_result_path = step_result_path / fs::path("joint_recon_" + to_string(iters_joint_optimization) + "_post");
        safe_create(joint_post_result_path);

        ceres::Problem problem;
        VectorXd params = param_sets[0].model.Wid;

        // Add constraints from each image
        for(auto i : consistent_set) {
          // Create a projected model first
          vector<MultilinearModel> model_projected_i(param_sets[i].indices.size());
          for(size_t j=0;j<param_sets[i].indices.size();++j) {
            model_projected_i[j] = model.project(vector<int>(1, param_sets[i].indices[j]));
            model_projected_i[j].ApplyWeights(param_sets[i].model.Wid, param_sets[i].model.Wexp);
          }

          // Create relevant matrices
          glm::dmat4 Rmat_i = glm::eulerAngleYXZ(param_sets[i].model.R[0], param_sets[i].model.R[1],
                                                 param_sets[i].model.R[2]);
          glm::dmat4 Tmat_i = glm::translate(glm::dmat4(1.0),
                                             glm::dvec3(param_sets[i].model.T[0],
                                                        param_sets[i].model.T[1],
                                                        param_sets[i].model.T[2]));
          glm::dmat4 Mview_i = Tmat_i * Rmat_i;

          double puple_distance = glm::distance(
            0.5 * (param_sets[i].recon.cons[28].data + param_sets[i].recon.cons[30].data),
            0.5 * (param_sets[i].recon.cons[32].data + param_sets[i].recon.cons[34].data));
          double weight_i = 100.0 / puple_distance;

          // Add per-vertex constraints
          for(size_t j=0;j<param_sets[i].indices.size();++j) {
            ceres::CostFunction * cost_function = new IdentityCostFunction_analytic(
              model_projected_i[j], param_sets[i].recon.cons[j], params.size(), Mview_i, Rmat_i,
              param_sets[i].cam, weight_i);

            problem.AddResidualBlock(cost_function, NULL, params.data());
          }
        }

        // Add prior constraint
        ceres::DynamicNumericDiffCostFunction<PriorCostFunction> *prior_cost_function =
          new ceres::DynamicNumericDiffCostFunction<PriorCostFunction>(
            new PriorCostFunction(prior.Wid_avg, prior.inv_sigma_Wid,
                                  prior.weight_Wid * consistent_set.size()));
        prior_cost_function->AddParameterBlock(params.size());
        prior_cost_function->SetNumResiduals(1);
        problem.AddResidualBlock(prior_cost_function, NULL, params.data());

        // Solve it
        {
          boost::timer::auto_cpu_timer timer_solve(
            "[Identity optimization] Problem solve time = %w seconds.\n");
          ceres::Solver::Options options;
          options.max_num_iterations = 3;
          options.minimizer_type = ceres::LINE_SEARCH;
          options.line_search_direction_type = ceres::LBFGS;
          DEBUG_EXPR(options.minimizer_progress_to_stdout = true;)
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          DEBUG_OUTPUT(summary.FullReport())
        }

        // Update the identity weights
        for(auto& param : param_sets) {
          param.model.Wid = params;

          // Also update geometry if needed
          {
            model.ApplyWeights(param.model.Wid, param.model.Wexp);
            param.mesh.UpdateVertices(model.GetTM());
            param.mesh.ComputeNormals();
          }
        }

        for(auto i : consistent_set) {
          if(true) {
            VisualizeReconstructionResult(joint_post_result_path, i);
          }
        }

        identity_weights_centroid_history.push_back(params);
      }
    }
  } // end of main reconstruction loop

  // Output the reconstructed identity weights
  {
    for(size_t i=0;i<identity_weights_history.size();++i) {
      ofstream fout("identity_matrix" + std::to_string(i) + ".txt");
      fout << identity_weights_history[i];
      fout.close();
    }

    for(size_t i=0;i<identity_weights_centroid_history.size();++i) {
      ofstream fout("identity_centroid" + std::to_string(i) + ".txt");
      fout << identity_weights_centroid_history[i];
      fout.close();
    }
  }

  // Output the chosen subset
  {
    ofstream fout( (result_path / fs::path("selection.txt")).string() );
    for(int i=0;i<final_chosen_set.size();++i) {
      // The row indices in the settings file!
      const int row_index = final_chosen_set[i];
      const int L = param_sets[row_index].img_filename.size();
      fout << param_sets[row_index].img_filename.substr(0, L-4) << endl;
    }
    fout.close();
  }

  // Visualize the final reconstruction results
  for(int i=0;i<num_images;++i) {
    // Visualize the reconstruction results
    #if 0
    MeshVisualizer* w = new MeshVisualizer("reconstruction result", param_sets[i].mesh);
    w->BindConstraints(image_points_pairs[i].second);
    w->BindImage(image_points_pairs[i].first);
    w->BindLandmarks(init_indices);

    w->BindUpdatedLandmarks(param_sets[i].indices);
    w->SetMeshRotationTranslation(param_sets[i].model.R, param_sets[i].model.T);
    w->SetCameraParameters(param_sets[i].cam);

    int show_width = image_points_pairs[i].first.width();
    int show_height = image_points_pairs[i].first.height();
    double show_ratio = 640.0 / show_height;
    w->resize(show_width * show_ratio, 640);
    w->show();

    w->paintGL();
    QImage recon_image = w->grabFrameBuffer();
    fs::path image_path = fs::path(image_filenames[i]);

    recon_image.save( (result_path / fs::path(image_path.stem().string() + "_recon.png")).string().c_str() );
    #else
    VisualizeReconstructionResult(result_path, i);
    #endif
    ofstream fout(image_filenames[i] + ".res");
    fout << param_sets[i].cam << endl;
    fout << param_sets[i].model << endl;
    fout << param_sets[i].stats << endl;
    fout.close();
  }

  return true;
}

#endif //MULTILINEARRECONSTRUCTION_MULTIIMAGERECONSTRUCTOR_H
