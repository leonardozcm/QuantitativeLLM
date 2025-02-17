#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace utils
{
    namespace post_process_utils
    {
        template <typename Serializable>
        void appendCSV(std::ofstream &file, const std::vector<Serializable> &row)
        {
            if (file.is_open())
            {
                for (size_t i = 0; i < row.size(); ++i)
                {
                    file << row[i];
                    if (i < row.size() - 1)
                    {
                        file << ",";
                    }
                }
                file << std::endl;
            }
            else
            {
                std::cerr << "Can't write to CSV file." << std::endl;
            }
        };
    }

    namespace string_utils
    {
        /**
         * Usage:
         *
         * ```cpp
         *  std::string s = utils::string_utils::string_format("Chunked  %d transpose", 1);
         * ```
         *
         * then you will get `s = "Chunked 1 transpose"; `.
         * **/
        template <typename... Args>
        std::string string_format(const std::string &format, Args... args)
        {
            int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
            if (size_s <= 0)
            {
                throw std::runtime_error("Error during formatting.");
            }
            auto size = static_cast<size_t>(size_s);
            std::unique_ptr<char[]> buf(new char[size]);
            std::snprintf(buf.get(), size, format.c_str(), args...);
            return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
        }

        std::string get_filename_without_extension(const std::string &path);

        std::string get_exec_func_name(const std::string &s);

        char* getCmdOption(char ** begin, char ** end, const std::string & option);
        
        #ifdef CUDA_EXECUTE
        void throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &command,
                 cudaError_t error_code);

        void exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                cudaError_t error_code);
        #endif

    }
};

namespace strutils = utils::string_utils;
namespace postutils = utils::post_process_utils;

#ifdef CUDA_EXECUTE

void utils::string_utils::throw_error(const std::string &filename,
                 std::size_t lineno,
                 const std::string &command,
                 cudaError_t error_code)
{
  throw std::runtime_error(string_format("%s:%d: Cuda API call returned error: "
                                       "%s: %s\nCommand: '%s'",
                                       filename.c_str(),
                                       lineno,
                                       cudaGetErrorName(error_code),
                                       cudaGetErrorString(error_code),
                                       command.c_str()));
}

void utils::string_utils::exit_error(const std::string &filename,
                std::size_t lineno,
                const std::string &command,
                cudaError_t error_code)
{
  std::cout<<string_format(
             "%s:%d: Cuda API call returned error: %s: %s\nCommand: '%s'\n",
             filename.c_str(),
             lineno,
             cudaGetErrorName(error_code),
             cudaGetErrorString(error_code),
             command.c_str());
  std::exit(EXIT_FAILURE);
}

#endif

std::string utils::string_utils::get_filename_without_extension(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string fileName = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);

    size_t lastDot = fileName.find_last_of('.');
    if (lastDot != std::string::npos) {
        fileName = fileName.substr(0, lastDot);
    }

    return fileName;
}

std::string utils::string_utils::get_exec_func_name(const std::string& s){
    if(s[0]=='('){
        auto func_name = s.substr(1, s.find_first_of(')')-1);
        for(int i=0; i<func_name.size(); i++){
            if(func_name[i]==',')func_name[i]='_';
        }
        return func_name;
    }else{
        return s.substr(0, s.find_first_of('('));
    }
}

char* utils::string_utils::getCmdOption(char ** begin, char ** end, const std::string & option)
{
    char ** itr = std::find(begin, end, option);
    if (itr != end && ++itr != end)
    {
        return *itr;
    }
    return 0;
}
#endif