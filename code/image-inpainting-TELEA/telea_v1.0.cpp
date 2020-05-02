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

const int To[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};

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
    int width = 50;
    int interval = 2;
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

priority_queue<BandPixel> NarrowBand;

void Init(){

    //initiate B

    for(int i = -epsilon; i <= epsilon; i ++)
        for(int j = -epsilon; j <= epsilon; j ++)
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

    // initiate f & T
    
    T = new float[N * M];
    memset(T, 0, sizeof(float) * N * M);

    f = new char[N * M];
    memset(f, KNOWN, sizeof(char) * N * M);

    for(int i = 0; i < N; i ++)
        for(int j = 0; j < M; j ++){
            if(mask.at<uchar>(i, j) == 0)
                continue;
            f[i * M + j] = INSIDE;
            T[i * M + j] = 1e6;
            for(int k = 0; k < 4; k ++){
                int ii = i + To[k][0], jj = j + To[k][1];
                if(Check(ii, jj) && f[ii * M + jj] == KNOWN)
                    f[ii * M + jj] = BAND;
            }
        }

}

pair<float, float> GradT(int x, int y){
    static pair<float, float> re;
    if(x + 1 >= N || f[(x + 1) * M + y] == INSIDE)
        if(x - 1 < 0 || f[(x - 1) * M + y] == INSIDE)
            re.X = 0;
        else
            re.X = (T[x * M + y] - T[(x - 1) * M + y]);
    else if(x - 1 < 0 || f[(x - 1) * M + y] == INSIDE)
        re.X = (T[(x + 1) * M + y] - T[x * M + y]);
    else
        re.X = (T[(x + 1) * M + y] - T[(x - 1) * M + y]) / 2;
    if(y + 1 >= M || f[x * M + y + 1] == INSIDE)
        if(y - 1 < 0 || f[x * M + y - 1] == INSIDE)
            re.Y = 0;
        else
            re.Y = (T[x * M + y] - T[x * M + y - 1]);
    else if(y - 1 < 0 || f[x * M + y - 1] == INSIDE)
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
        dst = 1.0 / (1 + r.X * r.X + r.Y * r.Y);
        lev = 1.0 / (1 + fabs(T[x * M + y] - T[i * M + j]));
        w = dir * dst * lev + 1e-6;

        gradI = GradI(i, j);
        Ia += w * ((Vec3f)img.at<Vec3b>(i, j) + (gradI.X * r.X + gradI.Y * r.Y));
        s += w;
    }
    img.at<Vec3b>(x, y) = (Vec3b)(Ia / s);
    if(img.at<Vec3b>(x, y) == Vec3b(0, 0, 0) || img.at<Vec3b>(x, y) == Vec3b(255, 255, 255)){
        cout << "Ia : " << Ia << endl;
        cout << "s : " << s << endl;
        cout << "img : " << img.at<Vec3b>(x, y) << endl;
    }
}

float solEqua(int i1, int j1, int i2, int j2){
    static float r, s, T1, T2;
    float re = 1e6;
    if(!Check(i1, j1) || !Check(i2, j2))
        return re;
    T1 = T[i1 * M + j1], T2 = T[i2 * M + j2];
    if(f[i1 * M + j1] == KNOWN){
        if(f[i2 * M + j2] == KNOWN){
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
    else if(f[i2 * M + j2] == KNOWN)
        re = 1 + T2;
    return re;
}

void Solve(){
    BandPixel t;
    int i, j;

    // initiate NarrowBand
    
    for(i = 0; i < N; i ++)
        for(j = 0; j < M; j ++)
            if(f[i * M + j] == BAND)
                NarrowBand.push(BandPixel(T[i * M + j], i, j));

    // inpaint each pixel
    
    while(!NarrowBand.empty()){
        t = NarrowBand.top();
        NarrowBand.pop();
        i = t.x, j = t.y;
        
        f[i * M + j] = KNOWN;
        for(int k = 0; k < 4; k ++){
            int ii = i + To[k][0], jj = j + To[k][0];
            if(!Check(ii, jj))
                continue;
            T[ii * M + jj] = min(min(solEqua(ii - 1, jj, ii, jj - 1),
                                     solEqua(ii + 1, jj, ii, jj - 1)),
                                 min(solEqua(ii - 1, jj, ii, jj + 1),
                                     solEqua(ii + 1, jj, ii, jj + 1)));
            if(f[ii * M + jj] == INSIDE){
                inpaint(ii, jj);
                f[ii * M + jj] = BAND;
                NarrowBand.push(BandPixel(T[ii * M + jj], ii, jj));
            }
        }
    }
    imwrite(output_path + "output.jpg", img);
    imshow("output", img);
    waitKey(0);
}

int main(){

    Init();
    Mat imgdst = img;

    inpaint(img, mask, imgdst, epsilon, INPAINT_TELEA);
    imshow("mask", mask);
    imshow("img", imgdst);
    waitKey(1);

    Solve();

    return 0;
}
