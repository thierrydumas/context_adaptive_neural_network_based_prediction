#include "tools.h"

bool is_string_special_characters_exclusively(const std::string& string_in)
{
    return string_in.find_first_not_of(" \t\f\v\n\r") == std::string::npos;
}

int parse_file_strings_one_key(std::map<unsigned int, std::string>& map_storage,
                               const std::string& path_to_file,
                               const std::string& delimiters)
{
    std::ifstream file_stream(path_to_file);
    std::string line_text("");
    std::vector<std::string> vector_substrings;
    std::string value_string("");
    if (file_stream.is_open())
    {
        while (file_stream.good())
        {
            std::getline(file_stream,
                         line_text);
            
            // A line containing special characters exclusively is ignored.
            if (is_string_special_characters_exclusively(line_text))
            {
                continue;
            }
            split_string(vector_substrings,
                         line_text,
                         delimiters);
            value_string = vector_substrings.at(1);
            remove_leading_trailing_whitespaces(value_string);
            map_storage[std::stoul(vector_substrings.at(0))] = value_string;
            
            /*
            If `vector_substrings` is not cleared, the substrings
            from the next loop iterations will be pushed back after
            the substrings from the previous loop iterations.
            */
            vector_substrings.clear();
        }
        file_stream.close();
        return 0;
    }
    else
    {
        fprintf(stderr, "The file at \"%s\" cannot be opened.\n", path_to_file.c_str());
        return -1;
    }
}

int parse_file_strings_three_keys(std::map<std::pair<unsigned int, unsigned int>, std::string>& map_false,
                                  std::map<std::pair<unsigned int, unsigned int>, std::string>& map_true,
                                  const std::string& path_to_file,
                                  const std::string& delimiters)
{
    std::ifstream file_stream(path_to_file);
    std::string line_text("");
    std::vector<std::string> vector_substrings;
    unsigned int key_unsigned_int_0(0);
    unsigned int key_unsigned_int_1(0);
    std::string value_string("");
    if (file_stream.is_open())
    {
        /*
        The method `std::ios::good` returns true if none
        of the stream's error state flags is set.
        */
        while (file_stream.good())
        {
            std::getline(file_stream,
                         line_text);
            
            // A line containing special characters exclusively is ignored.
            if (is_string_special_characters_exclusively(line_text))
            {
                continue;
            }
            split_string(vector_substrings,
                         line_text,
                         delimiters);
            
            /*
            `std::stoul` throws a `std::invalid_argument` exception
            if no conversion can be performed.
            `std::stoul` discards all the whitespace characters found
            in its input string, identified by calling `isspace`.
            */
            key_unsigned_int_0 = std::stoul(vector_substrings.at(0));
            key_unsigned_int_1 = std::stoul(vector_substrings.at(2));
            value_string = vector_substrings.at(3);
            remove_leading_trailing_whitespaces(value_string);
            if (std::stoul(vector_substrings.at(1)))
            {
                map_true[std::make_pair(key_unsigned_int_0, key_unsigned_int_1)] = value_string;
            }
            else
            {
                map_false[std::make_pair(key_unsigned_int_0, key_unsigned_int_1)] = value_string;
            }
            vector_substrings.clear();
        }
        file_stream.close();
        return 0;
    }
    else
    {
        fprintf(stderr, "The file at \"%s\" cannot be opened.\n", path_to_file.c_str());
        return -1;
    }
}

void remove_leading_trailing_whitespaces(std::string& string_to_be_modified)
{
    std::size_t index_position_first(string_to_be_modified.find_first_not_of(" \t\f\v\n\r"));
    
    /*
    The second argument of `std::string::erase` is optional.
    Its default value is `std::string::npos`, meaning that all
    the characters until the end of the string are erased.
    */
    string_to_be_modified.erase(0, index_position_first);
    std::size_t index_position_last(string_to_be_modified.find_last_not_of(" \t\f\v\n\r"));
    string_to_be_modified.erase(index_position_last + 1);
}

void split_string(std::vector<std::string>& vector_substrings,
                  const std::string& input_string,
                  const std::string& delimiters)
{
    const std::regex reg("[" + delimiters + "]+");
    
    /*
    `std::sregex_token_iterator` is defined as
    `std::regex_token_iterator<std::string::const_iterator>`.
    */
    std::sregex_token_iterator it(input_string.begin(),
                                  input_string.end(),
                                  reg,
                                  -1);
    const std::sregex_token_iterator it_end;
    
    /*
    Here, at each incrementation, the iterator advances to the
    next part of the string that is not matched by the given
    regular expression.
    */
    for (; it != it_end; it++)
    {
        vector_substrings.push_back(it->str());
    }
}


