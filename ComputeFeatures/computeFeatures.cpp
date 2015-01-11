/**
 * Code to compute CNN (ImageNet) features for a given image using CAFFE
 * (c) Rohit Girdhar
 */

#include <memory>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "caffe/caffe.hpp"
#include "utils.hpp"

#define FEAT_DIR fs::path("selsearch_feats")

using namespace std;
using namespace caffe;
using namespace cv;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

template<typename Dtype>
void read2DMatrixTxt(const fs::path&, vector<vector<Dtype>>&);
void sliceBoxes(const Mat&, const vector<vector<float>>&, vector<Mat>&);
void storeImages(const vector<Mat>&, const fs::path&);

int main(int argc, char *argv[]) {
    #ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
    LOG(INFO) << "Extracting Features in CPU mode";
    #else
    Caffe::set_mode(Caffe::GPU);
    #endif
    Caffe::set_phase(Caffe::TEST); // important, else will give random features
    
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show this help")
        ("network-path,n", po::value<string>()->required(),
         "Path to the prototxt file")
        ("model-path,m", po::value<string>()->required(),
         "Path to corresponding caffemodel")
        ("outdir,o", po::value<string>()->default_value("output"),
         "Output directory")
        ("layer,f", po::value<string>()->default_value("pool5"),
         "CNN layer to extract features from")
        ("imgsdir,i", po::value<string>()->required(),
         "Input directory of images")
        ("imgslist,l", po::value<string>()->required(),
         "File with list of images in imgsdir")
        ("tempdir,t", po::value<string>()->required(),
         "Path to where tempdata is stored, like selective search output")
        ("debug,d", po::bool_switch()->default_value(false),
         "Set debug, store image slices etc")
    ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
        LOG(INFO) << desc;
        return -1;
    }
    try {
        po::notify(vm);
    } catch(po::error& e) {
        LOG(ERROR) << e.what();
        return -1;
    }

    fs::path NETWORK_PATH = fs::path(vm["network-path"].as<string>());
    fs::path MODEL_PATH = 
        fs::path(vm["model-path"].as<string>());
    string LAYER = vm["layer"].as<string>();
    fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
    fs::path IMGSDIR = fs::path(vm["imgsdir"].as<string>());
    fs::path IMGSLIST = fs::path(vm["imgslist"].as<string>());
    fs::path TEMPDIR = fs::path(vm["tempdir"].as<string>());
    bool DEBUG = vm["debug"].as<bool>();

    NetParameter test_net_params;
    ReadProtoFromTextFile(NETWORK_PATH.string(), &test_net_params);
    Net<float> caffe_test_net(test_net_params);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(MODEL_PATH.string(), &trained_net_param);
    caffe_test_net.CopyTrainedLayersFrom(trained_net_param);
    int BATCH_SIZE = caffe_test_net.blob_by_name("data")->num();

    ifstream infile(IMGSLIST.c_str());
    string fname;
    int i = 0;

    // Compute features
    fs::create_directories(TEMPDIR / FEAT_DIR);
    while (infile >> fname) {
        i++;
        Mat I = imread((IMGSDIR / fs::path(fname)).string());
        vector<vector<float>> boxes;
        read2DMatrixTxt<float>((TEMPDIR / fs::path("selsearch_boxes") /
                    fs::path(to_string(i) + ".txt")), boxes);
        vector<Mat> slices;
        sliceBoxes(I, boxes, slices);
        transform(slices.begin(), slices.end(), slices.begin(),
                [](Mat I) -> Mat {
                    resize(I, I, Size(256, 256));
                    return I;
                });
        if (DEBUG) {
            LOG(INFO) << "DEBUG is set. Storing image slices";
            storeImages(slices, TEMPDIR / fs::path(string("slices_dump")) /
                    fs::path(to_string(i)));
        }
        vector<vector<float>> feats;
        computeFeatures<float>(caffe_test_net, slices, LAYER, BATCH_SIZE, feats);
        dumpFeats<float>(TEMPDIR / FEAT_DIR / 
                fs::path(to_string(i) + ".dat"), feats);
        LOG(INFO) << "Done for image " << i;
    }
    infile.close();

    return 0;
}

template<typename Dtype>
void read2DMatrixTxt(const fs::path& fpath, vector<vector<Dtype>>& output) {
    output.clear();
    ifstream infile(fpath.string().c_str());
    string line;
    while (getline(infile, line)) {
        string el;
        vector<Dtype> row;
        istringstream iss(line);
        while (getline(iss, el, ',')) {
            row.push_back((Dtype)stof(el));
        }
        output.push_back(row);
    }
    infile.close();
}

void sliceBoxes(const Mat& I, const vector<vector<float>>& boxes, vector<Mat>& slices) {
    // Strange as it may sound, SelectiveSearch returns boxes in y1,x1,y2,x2 format
    for (int i = 0; i < boxes.size(); i++) {
        float y = boxes[i][0] >= 0 ? boxes[i][0] : 0;
        float x = boxes[i][1] >= 0 ? boxes[i][1]: 0;
        float ymax = boxes[i][2] < I.rows ? boxes[i][2] : I.rows - 1;
        float xmax = boxes[i][3] < I.cols ? boxes[i][3] : I.cols - 1;
        Rect rect = Rect(x, y, xmax - x, ymax - y);
        slices.push_back(I(rect));
    }
}

void storeImages(const vector<Mat>& imgs, const fs::path& dpath) {
    // create directory if not exists
    fs::create_directories(dpath);
    for (int i = 0; i < imgs.size(); i++) {
        imwrite((dpath / fs::path(to_string(i) + ".jpg")).string(), imgs[i]);
    }
}

