#include <iostream>
#include <cuda_runtime.h>
//#include <cudnn.h>

using namespace std;

int main()
{
    int deviceCount = 0;
    int mpCount = 0;
    int smCount = 0;
    double fp32Perf = 0;
    int fp16mult = 0;

    cout << "yaTest/cuDnn" << endl <<  endl;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount > 0) {
        for (int i = 0; i < deviceCount; ++i) {
            cudaDeviceProp props{};
            cudaGetDeviceProperties(&props, i);

            double bandwidth = 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1024/1024;

            cout << " Currentt device:         " << i << endl;
            cout << " Device name:             " << props.name << endl;
            cout << " Memory size, MB:         " << props.totalGlobalMem/1024/1024 << endl;
            cout << " Memory Bandwidth, GB/s:  " << bandwidth << endl;
            cout << " Managed memory:          " << ((props.managedMemory>0)?"True":"False") << endl;
            cout << " CUDA version:            " << props.major << "." << props.minor << endl;
            cout << " Max clockrate, MHz:      " << props.clockRate/1000  << endl;

            mpCount = props.multiProcessorCount;
            switch(props.major){
                case 6: //Pascal
                    if(props.minor > 0)
                        smCount = mpCount*128;
                    else
                        smCount = mpCount*64;
                    fp16mult = 1;
                    break;
                case 7: //Volta/Turing
                        smCount = mpCount*64;
                    fp16mult = 2;
                    break;
                case 8: //Ampere
                    if(props.minor > 0)
                        smCount = mpCount*128;
                    else
                        smCount = mpCount*64;
                    fp16mult = 1;
                    break;
                case 9: //Hooper
                    if(props.minor == 0)
                        smCount = mpCount*64;
                    fp16mult = 1;
                    break;
                default:
                    cout << "No idea ;)" << std::endl;
                    break;
            }

            fp32Perf = smCount*props.clockRate*2.f/1000/1000;
            cout << " Perf FP32, GFLOPS:       " << fp32Perf  << endl;
            cout << " Perf FP16, GFLOPS:       " << fp32Perf*fp16mult  << endl;
            cout << " Perf FP64, GFLOPS:       " << fp32Perf/props.singleToDoublePrecisionPerfRatio  << endl;
        }
    } else {
        cout << "No CUDA device found" << endl;
    }

    return 0;
}
