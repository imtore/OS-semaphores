#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <sstream> //this header file is needed when using stringstream
#include <fstream>
#include <cstring>
#include <semaphore.h>
#include <thread>
#include <unistd.h>

#define MNIST_TESTING_SET_IMAGE_FILE_NAME "data/t10k-images-idx3-ubyte" ///< MNIST image testing file in the data folder
#define MNIST_TESTING_SET_LABEL_FILE_NAME "data/t10k-labels-idx1-ubyte" ///< MNIST label testing file in the data folder

#define HIDDEN_WEIGHTS_FILE "net_params/hidden_weights.txt"
#define HIDDEN_BIASES_FILE "net_params/hidden_biases.txt"
#define OUTPUT_WEIGHTS_FILE "net_params/out_weights.txt"
#define OUTPUT_BIASES_FILE "net_params/out_biases.txt"

#define NUMBER_OF_INPUT_CELLS 784  ///< use 28*28 input cells (= number of pixels per MNIST image)
#define NUMBER_OF_HIDDEN_CELLS 256 ///< use 256 hidden cells in one hidden layer
#define NUMBER_OF_OUTPUT_CELLS 10  ///< use 10 output cells to model 10 digits (0-9)
#define NUMBER_OF_HIDDEN_NEURONS_PER_THREAD 32
#define NUMBER_OF_HIDDEN_THREADS 8

#define MNIST_MAX_TESTING_IMAGES 10000 ///< number of images+labels in the TEST file/s
#define MNIST_IMG_WIDTH 28             ///< image width in pixel
#define MNIST_IMG_HEIGHT 28            ///< image height in pixel

using namespace std;

typedef struct MNIST_ImageFileHeader MNIST_ImageFileHeader;
typedef struct MNIST_LabelFileHeader MNIST_LabelFileHeader;

typedef struct MNIST_Image MNIST_Image;
typedef uint8_t MNIST_Label;
typedef struct Hidden_Node Hidden_Node;
typedef struct Output_Node Output_Node;
vector<Hidden_Node> hidden_nodes(NUMBER_OF_HIDDEN_CELLS);
vector<Output_Node> output_nodes(NUMBER_OF_OUTPUT_CELLS);

/**
 * @brief Data block defining a hidden cell
 */

struct Hidden_Node
{
    double weights[28 * 28];
    double bias;
    double output;
};

/**
 * @brief Data block defining an output cell
 */

struct Output_Node
{
    double weights[256];
    double bias;
    double output;
};

/**
 * @brief Data block defining a MNIST image
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_Image
{
    uint8_t pixel[28 * 28];
};

/**
 * @brief Data block defining a MNIST image file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first image.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_ImageFileHeader
{
    uint32_t magicNumber;
    uint32_t maxImages;
    uint32_t imgWidth;
    uint32_t imgHeight;
};

/**
 * @brief Data block defining a MNIST label file header
 * @attention The fields in this structure are not used.
 * What matters is their byte size to move the file pointer
 * to the first label.
 * @see http://yann.lecun.com/exdb/mnist/ for details
 */

struct MNIST_LabelFileHeader
{
    uint32_t magicNumber;
    uint32_t maxImages;
};

/**
 * @details Set cursor position to given coordinates in the terminal window
 */

void locateCursor(const int row, const int col)
{
    printf("%c[%d;%dH", 27, row, col);
}

/**
 * @details Clear terminal screen by printing an escape sequence
 */

void clearScreen()
{
    printf("\e[1;1H\e[2J");
}

/**
 * @details Outputs a 28x28 MNIST image as charachters ("."s and "X"s)
 */

void displayImage(MNIST_Image *img, int row, int col)
{
    char imgStr[(MNIST_IMG_HEIGHT * MNIST_IMG_WIDTH) + ((col + 1) * MNIST_IMG_HEIGHT) + 1];
    strcpy(imgStr, "");

    for (int y = 0; y < MNIST_IMG_HEIGHT; y++)
    {

        for (int o = 0; o < col - 2; o++)
            strcat(imgStr, " ");
        strcat(imgStr, "|");

        for (int x = 0; x < MNIST_IMG_WIDTH; x++)
        {
            strcat(imgStr, img->pixel[y * MNIST_IMG_HEIGHT + x] ? "X" : ".");
        }
        strcat(imgStr, "\n");
    }

    if (col != 0 && row != 0)
        locateCursor(row, 0);
    printf("%s", imgStr);
}

/**
 * @details Outputs a 28x28 text frame at a defined screen position
 */

void displayImageFrame(int row, int col)
{

    if (col != 0 && row != 0)
        locateCursor(row, col);

    printf("------------------------------\n");

    for (int i = 0; i < MNIST_IMG_HEIGHT; i++)
    {
        for (int o = 0; o < col - 1; o++)
            printf(" ");
        printf("|                            |\n");
    }

    for (int o = 0; o < col - 1; o++)
        printf(" ");
    printf("------------------------------");
}

/**
 * @details Outputs reading progress while processing MNIST testing images
 */

void displayLoadingProgressTesting(int imgCount, int y, int x)
{

    float progress = (float)(imgCount + 1) / (float)(MNIST_MAX_TESTING_IMAGES)*100;

    if (x != 0 && y != 0)
        locateCursor(y, x);

    printf("Testing image No. %5d of %5d images [%d%%]\n                                  ", (imgCount + 1), MNIST_MAX_TESTING_IMAGES, (int)progress);
}

/**
 * @details Outputs image recognition progress and error count
 */

void displayProgress(int imgCount, int errCount, int y, int x)
{

    double successRate = 1 - ((double)errCount / (double)(imgCount + 1));

    if (x != 0 && y != 0)
        locateCursor(y, x);

    printf("Result: Correct=%5d  Incorrect=%5d  Success-Rate= %5.2f%% \n", imgCount + 1 - errCount, errCount, successRate * 100);
}

/**
 * @details Reverse byte order in 32bit numbers
 * MNIST files contain all numbers in reversed byte order,
 * and hence must be reversed before using
 */

uint32_t flipBytes(uint32_t n)
{

    uint32_t b0, b1, b2, b3;

    b0 = (n & 0x000000ff) << 24u;
    b1 = (n & 0x0000ff00) << 8u;
    b2 = (n & 0x00ff0000) >> 8u;
    b3 = (n & 0xff000000) >> 24u;

    return (b0 | b1 | b2 | b3);
}

/**
 * @details Read MNIST image file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readImageFileHeader(FILE *imageFile, MNIST_ImageFileHeader *ifh)
{

    ifh->magicNumber = 0;
    ifh->maxImages = 0;
    ifh->imgWidth = 0;
    ifh->imgHeight = 0;

    fread(&ifh->magicNumber, 4, 1, imageFile);
    ifh->magicNumber = flipBytes(ifh->magicNumber);

    fread(&ifh->maxImages, 4, 1, imageFile);
    ifh->maxImages = flipBytes(ifh->maxImages);

    fread(&ifh->imgWidth, 4, 1, imageFile);
    ifh->imgWidth = flipBytes(ifh->imgWidth);

    fread(&ifh->imgHeight, 4, 1, imageFile);
    ifh->imgHeight = flipBytes(ifh->imgHeight);
}

/**
 * @details Read MNIST label file header
 * @see http://yann.lecun.com/exdb/mnist/ for definition details
 */

void readLabelFileHeader(FILE *imageFile, MNIST_LabelFileHeader *lfh)
{

    lfh->magicNumber = 0;
    lfh->maxImages = 0;

    fread(&lfh->magicNumber, 4, 1, imageFile);
    lfh->magicNumber = flipBytes(lfh->magicNumber);

    fread(&lfh->maxImages, 4, 1, imageFile);
    lfh->maxImages = flipBytes(lfh->maxImages);
}

/**
 * @details Open MNIST image file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st IMAGE
 */

FILE *openMNISTImageFile(char *fileName)
{

    FILE *imageFile;
    imageFile = fopen(fileName, "rb");
    if (imageFile == NULL)
    {
        printf("Abort! Could not find MNIST IMAGE file: %s\n", fileName);
        exit(0);
    }

    MNIST_ImageFileHeader imageFileHeader;
    readImageFileHeader(imageFile, &imageFileHeader);

    return imageFile;
}

/**
 * @details Open MNIST label file and read header info
 * by reading the header info, the read pointer
 * is moved to the position of the 1st LABEL
 */

FILE *openMNISTLabelFile(char *fileName)
{

    FILE *labelFile;
    labelFile = fopen(fileName, "rb");
    if (labelFile == NULL)
    {
        printf("Abort! Could not find MNIST LABEL file: %s\n", fileName);
        exit(0);
    }

    MNIST_LabelFileHeader labelFileHeader;
    readLabelFileHeader(labelFile, &labelFileHeader);

    return labelFile;
}

/**
 * @details Returns the next image in the given MNIST image file
 */

MNIST_Image getImage(FILE *imageFile)
{

    MNIST_Image img;
    size_t result;
    result = fread(&img, sizeof(img), 1, imageFile);
    if (result != 1)
    {
        printf("\nError when reading IMAGE file! Abort!\n");
        exit(1);
    }

    return img;
}

/**
 * @details Returns the next label in the given MNIST label file
 */

MNIST_Label getLabel(FILE *labelFile)
{

    MNIST_Label lbl;
    size_t result;
    result = fread(&lbl, sizeof(lbl), 1, labelFile);
    if (result != 1)
    {
        printf("\nError when reading LABEL file! Abort!\n");
        exit(1);
    }

    return lbl;
}

/**
 * @brief allocate weights and bias to respective hidden cells
 */

void allocateHiddenParameters()
{
    int idx = 0;
    int bidx = 0;
    ifstream weights(HIDDEN_WEIGHTS_FILE);
    for (string line; getline(weights, line);) //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 28 * 28; ++i)
        {
            in >> hidden_nodes[idx].weights[i];
        }
        idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for (string line; getline(biases, line);) //read stream line by line
    {
        stringstream in(line);
        in >> hidden_nodes[bidx].bias;
        bidx++;
    }
    biases.close();
}

/**
 * @brief allocate weights and bias to respective output cells
 */

void allocateOutputParameters()
{
    int idx = 0;
    int bidx = 0;
    ifstream weights(OUTPUT_WEIGHTS_FILE);     //"layersinfo.txt"
    for (string line; getline(weights, line);) //read stream line by line
    {
        stringstream in(line);
        for (int i = 0; i < 256; ++i)
        {
            in >> output_nodes[idx].weights[i];
        }
        idx++;
    }
    weights.close();

    ifstream biases(OUTPUT_BIASES_FILE);
    for (string line; getline(biases, line);) //read stream line by line
    {
        stringstream in(line);
        in >> output_nodes[bidx].bias;
        bidx++;
    }
    biases.close();
}

/**
 * @details The output prediction is derived by finding the maxmimum output value
 * and returning its index (=0-9 number) as the prediction.
 */

int getNNPrediction()
{

    double maxOut = 0;
    int maxInd = 0;

    for (int i = 0; i < NUMBER_OF_OUTPUT_CELLS; i++)
    {

        if (output_nodes[i].output > maxOut)
        {
            maxOut = output_nodes[i].output;
            maxInd = i;
        }
    }

    return maxInd;
}

// class AtomicWriter
// {
//     std::ostringstream st;

//   public:
//     template <typename T>
//     AtomicWriter &operator<<(T const &t)
//     {
//         st << t;
//         return *this;
//     }
//     ~AtomicWriter()
//     {
//         std::string s = st.str();
//         std::cerr << s;
//         //fprintf(stderr,"%s", s.c_str());
//         // write(2,s.c_str(),s.size());
//     }
// };

int order = 0;

sem_t input_used;
sem_t input_ready[NUMBER_OF_HIDDEN_THREADS];
sem_t h_output_used[8];
sem_t h_output_ready[10];
sem_t o_output_used[10];
sem_t o_output_ready;

MNIST_Image pimg;

void get_input()
{
    FILE *imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);

    for (int imgC = 0; imgC < MNIST_MAX_TESTING_IMAGES; imgC++)
    {
        for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
        {
            sem_wait(&input_used);
        }

        pimg = getImage(imageFile);

        // cout << "input\n";

        for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
        {
            sem_post(&input_ready[i]);
        }
    }
}

void compute_hidden_neurons(int start)
{
    for (int imgC = 0; imgC < MNIST_MAX_TESTING_IMAGES; imgC++)
    {
        sem_wait(&input_ready[start / NUMBER_OF_HIDDEN_NEURONS_PER_THREAD]);

        for (int i = 0; i < 10; i++)
        {
            sem_wait(&h_output_used[start / NUMBER_OF_HIDDEN_NEURONS_PER_THREAD]);
        }

        // cout << "hidden\n";

        for (int i = start; i < NUMBER_OF_HIDDEN_NEURONS_PER_THREAD + start; i++)
        {
            hidden_nodes[i].output = 0;
            for (int j = 0; j < NUMBER_OF_INPUT_CELLS; j++)
            {
                //cout << "imgCount: " << imgC << " pixel: " << pimg.pixel[j] << endl;
                hidden_nodes[i].output += pimg.pixel[j] * hidden_nodes[i].weights[j];
            }
            //cout << "hidden_num: " << start / 32 << ", output: " << hidden_nodes[i].output << endl;

            hidden_nodes[i].output += hidden_nodes[i].bias;
            hidden_nodes[i].output = (hidden_nodes[i].output >= 0) ? hidden_nodes[i].output : 0;
        }

        for (int i = 0; i < 10; i++)
        {
            sem_post(&h_output_ready[i]);
        }

        sem_post(&input_used);
    }
}

void compute_output_neurons(int n)
{

    for (int imgC = 0; imgC < MNIST_MAX_TESTING_IMAGES; imgC++)
    {

        for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
        {
            sem_wait(&h_output_ready[n]);
        }

        sem_wait(&o_output_used[n]);

        output_nodes[n].output = 0;

        for (int i = 0; i < NUMBER_OF_HIDDEN_CELLS; i++)
        {
            output_nodes[n].output += hidden_nodes[i].output * output_nodes[n].weights[i];
        }
        output_nodes[n].output += 1 / (1 + exp(-1 * output_nodes[n].output));

        sem_post(&o_output_ready);

        for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
        {
            sem_post(&h_output_used[i]);
        }
    }
}

void compute_result()
{
    FILE *imageFile = openMNISTImageFile(MNIST_TESTING_SET_IMAGE_FILE_NAME);
    FILE *labelFile = openMNISTLabelFile(MNIST_TESTING_SET_LABEL_FILE_NAME);

    int errCount = 0;

    for (int imgC = 0; imgC < MNIST_MAX_TESTING_IMAGES; imgC++)
    {
        for (int i = 0; i < 10; i++)
        {
            sem_wait(&o_output_ready);
        }

        displayLoadingProgressTesting(imgC, 5, 5);

        MNIST_Label lbl = getLabel(labelFile);

        MNIST_Image img = getImage(imageFile);

        displayImage(&img, 8, 6);

        int predictedNum = getNNPrediction();
        if (predictedNum != lbl)
            errCount++;

        printf("\n      Prediction: %d   Actual: %d ", predictedNum, lbl);

        displayProgress(imgC, errCount, 5, 66);

        //cout << predictedNum << endl;
        //sleep(1);
        for (int i = 0; i < 10; i++)
        {
            sem_post(&o_output_used[i]);
        }
    }
    cout << "Error : " << errCount << endl;

    fclose(imageFile);
    fclose(labelFile);
}
/**
 * @details test the neural networks to obtain its accuracy when classifying
 * 10k images.
 */
void testNN()
{
    // screen output for monitoring progress
    //displayImageFrame(7, 5);

    // number of incorrect predictions

    sem_init(&input_used, 0, 8);
    for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
        sem_init(&input_ready[i], 0, 0);

    for (int i = 0; i < 8; i++)
    {
        sem_init(&h_output_used[i], 0, 10);
    }

    for (int i = 0; i < 10; i++)
    {
        sem_init(&h_output_ready[i], 0, 0);
    }

    for (int i = 0; i < 10; i++)
    {
        sem_init(&o_output_used[i], 0, 1);
    }

    sem_init(&o_output_ready, 0, 0);

    vector<thread> hidden_threads;
    vector<thread> output_threads;

    thread input(get_input);

    int start = 0;
    for (int i = 0; i < NUMBER_OF_HIDDEN_THREADS; i++)
    {
        thread t(compute_hidden_neurons, start);
        hidden_threads.push_back(move(t));
        start += NUMBER_OF_HIDDEN_NEURONS_PER_THREAD;
    }

    for (int i = 0; i < NUMBER_OF_OUTPUT_CELLS; i++)
    {
        thread t(compute_output_neurons, i);
        output_threads.push_back(move(t));
    }

    thread result(compute_result);

    input.join();
    for (auto &&t : hidden_threads)
    {
        t.join();
    }
    for (auto &&t : output_threads)
    {
        t.join();
    }
    result.join();
}
int main(int argc, const char *argv[])
{

    // remember the time in order to calculate processing time at the end
    time_t startTime = time(NULL);

    // clear screen of terminal window
    clearScreen();
    printf("    MNIST-NN: a simple 2-layer neural network processing the MNIST handwriting images\n");

    // alocating respective parameters to hidden and output layer cells
    allocateHiddenParameters();
    allocateOutputParameters();

    //test the neural network
    testNN();

    locateCursor(38, 5);

    // calculate and print the program's total execution time
    time_t endTime = time(NULL);
    double executionTime = difftime(endTime, startTime);
    printf("\n    DONE! Total execution time: %.1f sec\n\n", executionTime);

    return 0;
}
