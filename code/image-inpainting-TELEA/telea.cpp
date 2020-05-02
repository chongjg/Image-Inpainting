#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/photo.hpp"

#include<cmath>
#include<queue>
#include<cstdio>
#include<cstring>
#include<cstdlib>
#include<iostream>
#include<algorithm>

using namespace std;
using namespace cv;

#define KNOWN 0
#define BAND 1
#define INSIDE 2

#define epsilon 6

#define X first
#define Y second

const int To[4][2] = {{1,0}, {-1,0}, {0,1}, {0,-1}};

int B[epsilon * epsilon * 4 + 1][2];
int Btop = -1;

string img_path = "/home/chongjg/Desktop/image-inpainting/image/";
string output_path = "/home/chongjg/Desktop/image-inpainting/output/";

Mat img, mask;

int N, M;

char* f;
float* T;
bool* vis;

bool Check(int i, int j){ return 0 <= i && i < N && 0 <= j && j < M; }

void create_mask(Mat &img, string &img_path){
    int width = 30;
    int interval = 3;
    mask = Mat(img.rows, img.cols, CV_8UC1);
    for(int i = 0; i < img.rows; i ++)
        for(int j = 0; j < img.cols; j ++)
            mask.at<uchar>(i, j) = (((i / width) % interval == 0) & ((j / width) % interval == 0)) * 255;
    imwrite(img_path + "mask.jpg", mask);
}

struct BandPixel{

    float T;
    int x, y;

    BandPixel(){}
    BandPixel(float T, int x, int y) : T(T), x(x), y(y) {}

};

bool operator < (const BandPixel &a, const BandPixel &b){return a.T > b.T;}

float solEqua(int i1, int j1, int i2, int j2){
    static float r, s, T1, T2;
    float re = 1e6;
    if(!Check(i1, j1) || !Check(i2, j2))
        return re;
    T1 = T[i1 * M + j1], T2 = T[i2 * M + j2];
    if(T1 < 1e6){
        if(T2 < 1e6){
            r = sqrt(2 - (T1 - T2) * (T1 - T2));
            s = (T1 + T2 - r) / 2;
            if(s >= max(T1, T2))
                re = s;
            else if(s + r >= max(T1, T2))
                re = s + r;
        }
        else
            re = 1 + T1;
    }
    else if(T2 < 1e6)
        re = 1 + T2;
    return re;
}

void TentFilter(){
    const int kernal[] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    const int idx[9][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,0}, {0,1}, {1,-1}, {1,0}, {1,1}};
    float* tmp = new float[N * M];
    memset(tmp, 0, sizeof(float) * N * M);
    int ii, jj, w;
    for(int i = 0; i < N; i ++)
        for(int j = 0; j < M; j ++){
            w = 0;
            for(int k = 0; k < 9; k ++){
                ii = i + idx[k][0];
                jj = j + idx[k][1];
                if(Check(ii, jj))
                    w += kernal[k], tmp[i * M + j] += kernal[k] * T[ii * M + jj];
            }
            tmp[i * M + j] /= w;
        }
    for(int i = 0; i < N * M; i ++)
        T[i] = tmp[i];
}

priority_queue<BandPixel> NarrowBand;
queue<pair<int, int> > ToBeInpainted;

void Init(){
    int i, j;
    
    //initiate B

    for(i = -epsilon; i <= epsilon; i ++)
        for(j = -epsilon; j <= epsilon; j ++)
            if(i * i + j * j <= epsilon * epsilon){
                Btop ++;
                B[Btop][0] = i;
                B[Btop][1] = j;
            }

    // input image & mask
    
    img = imread(img_path + "test.jpg");
    create_mask(img, img_path);

    N = img.rows;
    M = img.cols;

    // initiate f
    
    f = new char[N * M];
    memset(f, KNOWN, sizeof(char) * N * M);

    for(i = 0; i < N; i ++)
        for(j = 0; j < M; j ++){
            if(mask.at<uchar>(i, j) == 0)
                continue;
            f[i * M + j] = INSIDE;
            for(int k = 0; k < 4; k ++){
                int ii = i + To[k][0], jj = j + To[k][1];
                if(Check(ii, jj) && f[ii * M + jj] == KNOWN)
                    f[ii * M + jj] = BAND;
            }
        }

    // initiate NarrowBand & T

    BandPixel t;

    T = new float[N * M];
    memset(T, 0, sizeof(float) * N * M);

    for(i = 0; i < N; i ++)
        for(j = 0; j < M; j ++)
            if(f[i * M + j] == BAND)
                NarrowBand.push(BandPixel(T[i * M + j], i, j));
            else
                T[i * M + j] = 1e6;

    bool *vis = new bool[N * M];
    memset(vis, false, sizeof(bool) * N * M);

    while(!NarrowBand.empty()){
        t = NarrowBand.top();
        NarrowBand.pop();
        i = t.x, j = t.y;
        if(vis[i * M + j])
            continue;
        vis[i * M + j] = true;
        ToBeInpainted.push(make_pair(i, j));
        for(int k = 0; k < 4; k ++){
            int ii = i + To[k][0], jj = j + To[k][1];
            if(!Check(ii, jj))
                continue;
            float tmpT = min(min(solEqua(ii - 1, jj, ii, jj - 1),
                                 solEqua(ii + 1, jj, ii, jj - 1)),
                             min(solEqua(ii - 1, jj, ii, jj + 1),
                                 solEqua(ii + 1, jj, ii, jj + 1)));
            tmpT = min(tmpT, T[i * M + j] + 1);
            if(tmpT < T[ii * M + jj]){
                T[ii * M + jj] = tmpT;
                NarrowBand.push(BandPixel(T[ii * M + jj], ii, jj));
            }
        }
    }

    for(i = 0; i < N; i ++)
        for(j = 0; j < M; j ++)
            if(f[i * M + j] == KNOWN)
                T[i * M + j] *= -1;
    
    TentFilter();
    
}

pair<float, float> GradT(int x, int y){
    static pair<float, float> re;
    if(x + 1 >= N)
        re.X = (T[x * M + y] - T[(x - 1) * M + y]);
    else if(x - 1 < 0)
        re.X = (T[(x + 1) * M + y] - T[x * M + y]);
    else
        re.X = (T[(x + 1) * M + y] - T[(x - 1) * M + y]) / 2;
    if(y + 1 >= M)
        re.Y = (T[x * M + y] - T[x * M + y - 1]);
    else if(y - 1 < 0)
        re.Y = (T[x * M + y + 1] - T[x * M + y]);
    else
        re.Y = (T[x * M + y + 1] - T[x * M + y - 1]) / 2;
    return re;
}

pair<Vec3f, Vec3f> GradI(int x, int y){
    static pair<Vec3f, Vec3f> re;
    for(int k = 0; k < 3; k ++){
        if(x + 1 >= N || f[(x + 1) * M + y] == INSIDE)
            if(x - 1 < 0 || f[(x - 1) * M + y] == INSIDE)
                re.X[k] = 0;
            else
                re.X[k] = ((float)img.at<Vec3b>(x, y)[k] - img.at<Vec3b>(x - 1, y)[k]);
        else if(x - 1 < 0 || f[(x - 1) * M + y] == INSIDE)
            re.X[k] = ((float)img.at<Vec3b>(x + 1, y)[k] - img.at<Vec3b>(x, y)[k]);
        else
            re.X[k] = ((float)img.at<Vec3b>(x + 1, y)[k] - img.at<Vec3b>(x - 1, y)[k]) / 2;
        if(y + 1 >= M || f[x * M + y + 1] == INSIDE)
            if(y - 1 < 0 || f[x * M + y - 1] == INSIDE)
                re.Y[k] = 0;
            else
                re.Y[k] = ((float)img.at<Vec3b>(x, y)[k] - img.at<Vec3b>(x, y - 1)[k]);
        else if(y - 1 < 0 || f[x * M + y - 1] == INSIDE)
            re.Y[k] = ((float)img.at<Vec3b>(x, y + 1)[k] - img.at<Vec3b>(x, y)[k]);
        else
            re.Y[k] = ((float)img.at<Vec3b>(x, y + 1)[k] - img.at<Vec3b>(x, y - 1)[k]) / 2;
    }
    return re;
}

void inpaint(int x, int y){
    static int i, j;
    static pair<int, int> r;
    static pair<float, float> gradT;
    static pair<Vec3f, Vec3f> gradI;
    static float dir, dst, lev, w;
    Vec3f Ia(0, 0, 0);
    float s = 0;
    gradT = GradT(x, y);
    for(int t = 0; t <= Btop; t ++){
        i = x + B[t][0], j = y + B[t][1];
        if(!Check(i, j) || f[i * M + j] == INSIDE)
            continue;
        r = make_pair(B[t][0], B[t][1]);
        dir = fabs(r.X * gradT.X + r.Y * gradT.Y) / sqrt(r.X * r.X + r.Y * r.Y);
        dst = 1.0 / (r.X * r.X + r.Y * r.Y);
        lev = 1.0 / (1 + fabs(T[x * M + y] - T[i * M + j]));
        w = dir * dst * lev;

        gradI = GradI(i, j);
        Ia += w * ((Vec3f)img.at<Vec3b>(i, j) + (gradI.X * r.X + gradI.Y * r.Y));
        s += w;
    }
    img.at<Vec3b>(x, y) = (Vec3b)(Ia / s);
}

void Solve(){
    int i, j;
    pair<int, int> p;
    
    while(!ToBeInpainted.empty()){
        p = ToBeInpainted.front();
        ToBeInpainted.pop();
        i = p.X, j = p.Y;
        f[i * M + j] = KNOWN;
        for(int k = 0; k < 4; k ++){
            int ii = i + To[k][0], jj = j + To[k][0];
            if(Check(ii, jj) && f[ii * M + jj] == INSIDE){
                inpaint(ii, jj);
                f[ii * M + jj] = BAND;
            }
        }
    }
    
}

int main(){

    Init();
    Solve();

    imshow("output", img);
    imwrite(output_path + "inpainted.jpg", img);
    waitKey(1);

    Mat imgdst(img.rows, img.cols, CV_8UC3);

    inpaint(img, mask, imgdst, epsilon, INPAINT_TELEA);
    imshow("cv_inpaint", imgdst);
    imwrite(output_path + "cv_inpaint.jpg", imgdst);
    waitKey(0);

    return 0;
}
