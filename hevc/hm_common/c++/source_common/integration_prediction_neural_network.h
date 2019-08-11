#ifndef INTEGRATION_PREDICTION_NEURAL_NETWORK_H
#define INTEGRATION_PREDICTION_NEURAL_NETWORK_H

/*
In <Windows.h>, there exist macros redefining min and max. These
macros are disabled via `NOMINMAX`.
*/
#ifdef _WIN32
    #define NOMINMAX
#endif

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

/** @brief Creates two tensors, each storing a masked context portion, for each width of target patch in {16, 32, 64} pixels.
  *
  * @details For each width of target patch in {16, 32, 64} pixels, the 1st tensor
  *          stores the masked context portion located above the target patch. The
  *          2nd tensor stores the masked context portion located on the left side
  *          of the target patch.
  *
  * @param tensors_portion_above Vector storing three masked context portions
  *                              located above the target patch, each for a width
  *                              of target patch in {16, 32, 64} pixels.
  * @param tensors_portion_left Vector storing three masked context portions
  *                             located on the left side of the target patch,
  *                             each for a width of target patch in {16, 32, 64}
  *                             pixels.
  *
  */
void create_tensors_context_portion(std::vector<tensorflow::Tensor>& tensors_portion_above,
                                    std::vector<tensorflow::Tensor>& tensors_portion_left);

/** @brief Creates a tensor storing a flattened masked context, for each width of target patch in {4, 8} pixels.
  *
  * @param tensors_flattened_context Vector storing the flattened masked context
  *                                  for each width of target patch in {4, 8} pixels.
  *
  */
void create_tensors_flattened_context(std::vector<tensorflow::Tensor>& tensors_flattened_context);

/** @brief Loads a graph and restores the parameters from a binary proto file.
  *
  * @param path_to_graph_output Path to the binary proto file containing the graph
  *                             to be loaded and the parameters.
  * @param ptr_session Pointer to the current session.
  * @return Status of the graph.
  *
  */
tensorflow::Status load_graph(const tensorflow::string& path_to_graph_output,
                              std::unique_ptr<tensorflow::Session>& unique_ptr_session);

/** @brief Loads multiple graphs and restores their parameters from a binary proto file.
  *
  * @param vector_paths_to_graphs_output Vector storing paths to binary proto files, each
  *                                      containing a graph to be loaded and its parameters.
  * @param vector_unique_ptrs_session Vector of pointers to sessions.
  * @return Status of the graphs.
  *
  */
tensorflow::Status load_graphs(const std::vector<std::string>& vector_paths_to_graphs_output,
                               std::vector<std::unique_ptr<tensorflow::Session>>& vector_unique_ptrs_session);

#endif


