#include <memory>
#include <fstream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp> // for to_lower
#include "utils.hpp"
#include "lock.hpp"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

void computeDistances(int, const vector<fs::path>&, vector<float>&, vector<int>&);

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "Show this help")
        ("outdir,o", po::value<string>()->default_value("output"),
         "Output directory")
    ;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
        LOG(ERROR) << desc;
        return -1;
    }
    try {
        po::notify(vm);
    } catch(po::error& e) {
        LOG(ERROR) << e.what();
        return -1;
    }

    fs::path OUTDIR = fs::path(vm["outdir"].as<string>());
    fs::path TEMPDIR = fs::path("../tempdata");

    vector<fs::path> imgs;
    readList<fs::path>(fs::path("../dataset/PeopleAtLandmarks/ImgsList.txt"), imgs);
    for (int i = 1; i <= 237; i++) {
        if (!lock("../tempdata/dists/" + to_string(i) + ".txt")) {
            LOG(ERROR) << "Already done for " << i;
            continue;
        }
        vector<float> dists;
        vector<int> bboxes;
        computeDistances(i, imgs, dists, bboxes);
        writeList<float>(string("../tempdata/dists/") + to_string(i) + ".txt", dists);
        writeList<int>(string("../tempdata/dists/") + to_string(i) + "_posn.txt", bboxes);
        LOG(ERROR) << "Done for " << i;
        unlock("../tempdata/dists/" + to_string(i) + ".txt");
    }

    return 0;
}

void computeDistances(int idx, const vector<fs::path>& imgslist, 
        vector<float>& dists, vector<int>& bboxes) {
    dists.clear(); bboxes.clear();
    vector<vector<float>> query_feats;
    loadFeats("../tempdata/marked_feats/" + to_string(idx) + ".dat", query_feats);
    // currently only supporting one feature in query
    CHECK_EQ(query_feats.size(), 1);
    for (int i = 1; i <= imgslist.size(); i++) {
        vector<vector<float>> test_feats;
        loadFeats("../tempdata/selsearch_feats/" + to_string(i) + ".dat", test_feats);
        vector<float> test_dists;
        float min_cos_dist = 1.0;
        int min_cos_dist_pos = 0;
        for (int j = 0; j < test_feats.size(); j++) {
            float d = cosine_distance(query_feats[0], test_feats[j]);
            if (min_cos_dist > d) {
                min_cos_dist = d;
                min_cos_dist_pos = j; // change to j+1, since 1 index is conven
            }
        }
        dists.push_back(min_cos_dist);
        bboxes.push_back(min_cos_dist_pos);
    }
}

