#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "caffe/caffe.hpp"

using namespace std;
using namespace caffe;
using namespace cv;
namespace fs = boost::filesystem;

// Output file style
#define DUMP_TXT 1
#define DUMP_BIN 2

template<typename Dtype>
void computeFeatures(Net<Dtype>& caffe_test_net,
        const vector<Mat>& imgs,
        string LAYER,
        int BATCH_SIZE,
        vector<vector<Dtype>>& output) {
    int nImgs = imgs.size();
    int nBatches = ceil(nImgs * 1.0f / BATCH_SIZE);
    for (int batch = 0; batch < nBatches; batch++) {
        int actBatchSize = min(nImgs - batch * BATCH_SIZE, BATCH_SIZE);
        vector<Mat> imgs_b;
        if (actBatchSize >= BATCH_SIZE) {
            imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, 
                    imgs.begin() + (batch + 1) * BATCH_SIZE);
        } else {
            imgs_b.insert(imgs_b.end(), imgs.begin() + batch * BATCH_SIZE, imgs.end());
            for (int j = actBatchSize; j < BATCH_SIZE; j++)
                imgs_b.push_back(imgs[0]);
        }
        vector<int> dvl(BATCH_SIZE, 0);
        boost::dynamic_pointer_cast<caffe::MemoryDataLayer<Dtype>>(
                caffe_test_net.layers()[0])->AddMatVector(imgs_b, dvl);
        vector<Blob<Dtype>*> dummy_bottom_vec;
        Dtype loss = 0.0f;
        caffe_test_net.ForwardPrefilled(&loss);
        const boost::shared_ptr<Blob<Dtype>> feat = caffe_test_net.blob_by_name(LAYER);
        for (int i = 0; i < actBatchSize; i++) {
            Dtype* feat_data = feat->mutable_cpu_data() + feat->offset(i);
            output.push_back(vector<Dtype>(feat_data, feat_data + feat->count() / feat->num()));
        }
        LOG(INFO) << "Batch " << batch << "/" << nBatches << " (" << actBatchSize << " images) done";
    }
}

/**
  * Function to return list of images in a directory (searched recursively).
  * The output paths are w.r.t. the path imgsDir
  */
void genImgsList(const fs::path& imgsDir, vector<fs::path>& list) {
    if(!fs::exists(imgsDir) || !fs::is_directory(imgsDir)) return;
    vector<string> imgsExts = {".jpg", ".png", ".jpeg", ".JPEG", ".PNG", ".JPG"};

    fs::recursive_directory_iterator it(imgsDir);
    fs::recursive_directory_iterator endit;
    while(it != endit) {
        if(fs::is_regular_file(*it) && 
                find(imgsExts.begin(), imgsExts.end(), 
                    it->path().extension()) != imgsExts.end())
            // write out paths but clip out the initial relative path from current dir 
            list.push_back(fs::path(it->path().relative_path().string().
                    substr(imgsDir.relative_path().string().length())));
        ++it;
    }
    LOG(INFO) << "Found " << list.size() << " image file(s) in " << imgsDir;
}

template<typename Dtype>
void dumpFeatsTxt(const fs::path& fpath, const vector<vector<Dtype>>& feats) {
    ofstream ofs(fpath.string());
    for (int i = 0; i < feats.size(); i++) {
        for (int j = 0; j < feats[i].size(); j++) {
            ofs << feats[i][j] << " ";
        }
        ofs << endl;
    }
    ofs.close();
}

template<typename Dtype>
void dumpFeats(const fs::path& fpath, const vector<vector<Dtype>>& feats,
        int type = DUMP_BIN) {
    if (type == DUMP_BIN) {
        ofstream of(fpath.string(), ios::binary);
        boost::archive::binary_oarchive ar(of);
        ar << feats;
        of.close();
    } else {
        // DUMP_TXT
        dumpFeatsTxt<Dtype>(fpath, feats);
    }
    // compress file, and delete this one
    fs::path dpath = fpath.parent_path();
    fs::path fname = fpath.filename();
    string cmd = string("cd ") + dpath.string() + "; " + 
        string("tar zcvpf ") + fname.string() + ".tar.gz " + fname.string() + "; " +
        string("rm ") + fname.string();
    system(cmd.c_str());
}

template<typename Dtype>
void loadFeats(const fs::path& fpath, vector<vector<Dtype>>& feats) {
    // decompress file, read and delete decompressed version
    fs::path dpath = fpath.parent_path();
    fs::path fname = fpath.filename();

    string cmd = string("cd ") + dpath.string() + "; " + 
        string("tar xzf ") + fname.string() + ".tar.gz";
    system(cmd.c_str());
    ifstream ifs(fpath.string(), ios::binary);
    boost::archive::binary_iarchive ar(ifs);
    ar >> feats;
    cmd = string("rm ") + fpath.string();
    system(cmd.c_str());
}

template<typename Dtype>
void readList(const fs::path& fpath, vector<Dtype>& lst) {
    lst.clear();
    ifstream ifs(fpath.string());
    Dtype el;
    while (ifs >> el) {
        lst.push_back(el);
    }
    ifs.close();
}

template<typename Dtype>
void writeList(const fs::path& fpath, const vector<Dtype>& lst) {
    ofstream ofs(fpath.string());
    for (int i = 0; i < lst.size(); i++) {
        ofs << lst[i] << endl;
    }
    ofs.close();
}

float norm_l2(const vector<float>& a) {
    float sqnorm = 0;
    for (int i = 0; i < a.size(); i++) {
        sqnorm += a[i] * a[i];
    }
    return sqrt(sqnorm);
}

float cosine_distance(const vector<float>& a, const vector<float>& b) {
    CHECK_EQ(a.size(), b.size());
    float sim = 0;
    for (int i = 0; i < a.size(); i++) {
        sim += a[i] * b[i];
    }
    sim = sim / norm_l2(a);
    sim = sim / norm_l2(b);
    return 1 - sim;
}

#endif

