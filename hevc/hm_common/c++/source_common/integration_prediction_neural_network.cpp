#include "integration_prediction_neural_network.h"

void create_tensors_context_portion(std::vector<tensorflow::Tensor>& tensors_portion_above,
                                    std::vector<tensorflow::Tensor>& tensors_portion_left)
{
    /*
    For each width of target patch, the masked context portion located
    above the target patch is put into `tensors_portion_above`.
    */
    tensors_portion_above.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 16, 48, 1}));
    tensors_portion_above.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 32, 96, 1}));
    tensors_portion_above.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 64, 192, 1}));
    
    /*
    For each width of target patch, the masked context portion located on
    the left side of the target patch is put into `tensors_portion_left`.
    */
    tensors_portion_left.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 32, 16, 1}));
    tensors_portion_left.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 64, 32, 1}));
    tensors_portion_left.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 128, 64, 1}));
}

void create_tensors_flattened_context(std::vector<tensorflow::Tensor>& tensors_flattened_context)
{
    tensors_flattened_context.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 80}));
    tensors_flattened_context.push_back(tensorflow::Tensor(tensorflow::DT_FLOAT, {1, 320}));
}

tensorflow::Status load_graph(const tensorflow::string& path_to_graph_output,
                              std::unique_ptr<tensorflow::Session>& unique_ptr_session)
{
    tensorflow::GraphDef graph_def;
    tensorflow::Status status_reading_graph(ReadBinaryProto(tensorflow::Env::Default(), path_to_graph_output, &graph_def));
    if (!status_reading_graph.ok())
    {
        return tensorflow::errors::NotFound("The graph at \"", path_to_graph_output, "\" cannot be loaded.");
    }
    
    /*
    `std::unique_ptr::reset`:
        (i) saves a copy of the current pointer.
        (ii) overwrites the current pointer.
        (iii) if the old pointer was non-empty, deletes the
        previously managed object.
    */
    unique_ptr_session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    tensorflow::Status status_creating_session(unique_ptr_session->Create(graph_def));
    if (!status_creating_session.ok())
    {
        return status_creating_session;
    }
    return tensorflow::Status::OK();
}

tensorflow::Status load_graphs(const std::vector<std::string>& vector_paths_to_graphs_output,
                               std::vector<std::unique_ptr<tensorflow::Session>>& vector_unique_ptrs_session)
{
    tensorflow::Status status_load_graph;
    for (unsigned int i(0); i < vector_paths_to_graphs_output.size(); i++)
    {
        status_load_graph = load_graph(vector_paths_to_graphs_output.at(i),
                                       vector_unique_ptrs_session.at(i));
        if (!status_load_graph.ok())
        {
            return status_load_graph;
        }
    }
    return tensorflow::Status::OK();
}


