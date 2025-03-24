#ifndef QTT_H
#define QTT_H

#include <cstdint>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include "utils.h"

#ifdef CUDA_EXECUTE
#include "qtt_cuda_runtime.cuh"
#include "qtt_cuda_stream.cuh"
#include "qtt_cuda_call.cuh"
#endif

#include "qtt_data_type.cuh"

#define TIME_ESITIMATE_FAIL -1
#define QTTBENCH_ASSERT(ASSERT_EXPR) \
    if(!ASSERT_EXPR) { \
        std::cerr <<"\033[31m"<< "Assert" << #ASSERT_EXPR << " fail at "<< __FILE__ << ":" << __LINE__ <<"\033[0m"<< std::endl; \
    }

// redirect to other stdout
#define QTTBENCH_LOG(loginfo) \
    std::cout<<"\033[32m"<<loginfo<<"\033[0m"<<std::endl;


#define QTTBENCH_WARN(loginfo) \
    std::cout<<"\033[33m"<<loginfo<<"\033[0m" <<std::endl;


#define QTTBENCH_HIGH_LIGHT(loginfo) \
    std::cout<<"\033[35m"<<loginfo<<"\033[0m" <<std::endl;


#define QTTBENCH_EXECUTE_TASK(exec_func) \
    strutils::get_exec_func_name(#exec_func), [&](){exec_func;}


#define QTTBENCH_VERIFY_TASK(veri_func) \
    [&](){return veri_func;}

// This macro is used to record function name automatically
#define QTTBENCH_RUN(state, exec_func, veri_func, args)\
    state.run(QTTBENCH_EXECUTE_TASK(exec_func(args)), QTTBENCH_VERIFY_TASK(verify(TRANSPOSE_ARGS)));

namespace qttbench{
    using time_t = std::int64_t;
    class State{
        int trials;
        bool no_warmup_;
        std::vector<time_t> estiminated_duration;
        std::vector<std::string> estiminated_tasks;
        std::ofstream csv_output;
        bool header_init_;
        bool perf_mode_=1;
        
        #ifdef CUDA_EXECUTE
        cuda_stream cuda_stream_;
        #endif

        void _write_header(){
            if(!header_init_){
                postutils::appendCSV(csv_output, estiminated_tasks);
                estiminated_tasks.clear();
                header_init_=true;
            }
        }

        std::pair<time_t, const char*> get_proper_time_present(time_t during){
            if(during>1e9){
                return std::make_pair(during/1e9, "s");
            }else if(during>1e6){
                return std::make_pair(during/1e6, "ms");
            }else if(during>1e3){
                return std::make_pair(during/1e3, "us");
            }else{
                return std::make_pair(during, "ns");
            }
        }

        public:
            // move only
            State(const State&)=delete;
            State(State&& )=default;
            State operator=(const State&)=delete;
            State &operator=(State&& )=default;

            State(int trials):trials(trials), header_init_(false), no_warmup_(false){
                QTTBENCH_ASSERT(trials>0);
            }

            State(int trials, bool no_warmup):trials(trials), header_init_(false), no_warmup_(no_warmup){
                QTTBENCH_ASSERT(trials>0);
            }

            State(int trials, bool no_warmup, bool perf_mode):trials(trials), header_init_(false), no_warmup_(no_warmup), perf_mode_(perf_mode){
                QTTBENCH_ASSERT(trials>0);
            }

            ~State(){
                if(csv_output.is_open()){
                    csv_output.flush();
                    csv_output.close();
                }
            }

            void set_csv_output(const std::string &filename="benchmark_result"){
                csv_output.open(filename+".csv", std::ios::app);
                QTTBENCH_ASSERT(csv_output.is_open());
            }


            template<typename ExecuteType, typename VerifyType>
            void warmup(std::string exec_name, ExecuteType exec_func, VerifyType exec_verify){
                QTTBENCH_LOG("Warm up start...");
                auto start = std::chrono::high_resolution_clock::now();

            #ifdef CUDA_EXECUTE
                for(int i=0; i<trials; i++){
                    exec_func(cuda_stream_.get());
                }
                QTTBENCH_CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
            #else
                for(int i=0; i<trials; i++){
                    exec_func();
                }
            #endif

                auto end = std::chrono::high_resolution_clock::now();
                QTTBENCH_LOG("Warm up end...");
                
                if(perf_mode_){
                    QTTBENCH_LOG("Verify correctness...");
                    
                    #ifdef CUDA_EXECUTE
                        QTTBENCH_ASSERT(exec_verify(cuda_stream_.get()));
                        QTTBENCH_CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
                    #else
                        QTTBENCH_ASSERT(exec_verify());
                    #endif
                }
                
                time_t during_t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                auto [during, units_t] = get_proper_time_present(during_t);
                QTTBENCH_LOG(strutils::string_format("Benchmark %s warms up within %d %s.", exec_name.c_str(), during, units_t));
            }
            
            template<typename ExecuteType, typename VerifyType>
            void execute(std::string exec_name, ExecuteType exec_func, VerifyType exec_verify){
                QTTBENCH_LOG("Benchmark start...");
                auto start = std::chrono::high_resolution_clock::now();

                #ifdef CUDA_EXECUTE
                    for(int i=0; i<trials; i++){
                        exec_func(cuda_stream_.get());
                    }
                    QTTBENCH_CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()))
                #else
                    for(int i=0; i<trials; i++){
                        exec_func();
                    }
                #endif

                auto end = std::chrono::high_resolution_clock::now();
                QTTBENCH_LOG("Benchmark end...");

                if(perf_mode_){
                    QTTBENCH_LOG("Verify correctness...");

                    #ifdef CUDA_EXECUTE
                        QTTBENCH_ASSERT(exec_verify(cuda_stream_.get()));
                        QTTBENCH_CUDA_CALL(cudaStreamSynchronize(cuda_stream_.get()));
                    #else
                        QTTBENCH_ASSERT(exec_verify());
                    #endif
                }

                time_t during_t = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                auto [during, units_t] = get_proper_time_present(during_t);
                QTTBENCH_LOG(strutils::string_format("Benchmark %s takes %d %s with %d times run, %d  %s per run.\n", exec_name.c_str(), during, units_t, trials, (time_t)(during/trials), units_t));
                estiminated_tasks.emplace_back(exec_name.c_str());
                estiminated_duration.emplace_back(during/trials);
            }

            template<typename ExecuteType, typename VerifyType>
            void run(std::string exec_name, ExecuteType exec_func, VerifyType exec_verify){
                QTTBENCH_HIGH_LIGHT(strutils::string_format("TASK << %s >> :", exec_name.c_str()));
                #ifdef CUDA_EXECUTE
                if(no_warmup_){
                    this->warmup(exec_name, exec_func, exec_verify);
                }
                #endif
                this->execute(exec_name, exec_func, exec_verify);
                std::cout.flush();
            }

            void dump_csv(){
                if(csv_output.is_open()){
                    _write_header();
                    postutils::appendCSV(csv_output, estiminated_duration);
                    QTTBENCH_HIGH_LIGHT("Benchmark results have been dumped to CSV file.")
                    estiminated_duration.clear();
                }else{
                    QTTBENCH_WARN("please call state.set_out_put() before you want to dump results.")
                }
            }
    };
}

#endif