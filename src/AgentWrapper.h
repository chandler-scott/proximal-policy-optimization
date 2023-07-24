/*
 * AgentWrapper.h
 *
 *  Created on: Jul 22, 2023
 *      Author: Chandler Scott
 */

#ifndef IOV_SIM_UTIL_AGENTWRAPPER_H_
#define IOV_SIM_UTIL_AGENTWRAPPER_H_

#include "PythonWrapper.h"
#include <unordered_map>
#include <string>

using namespace std;

class AgentWrapper
{
public:
    AgentWrapper();
    ~AgentWrapper();

    void loadStateDicts(PyObject *pStateDict, PyObject *vStateDict);

    std::pair<PyObject *, PyObject *> getStateDictsAsJson();
    std::pair<PyObject *, PyObject *> getStateDictsFromJson(PyObject *pBytes, PyObject *vBytes);

    PyObject *stringToPyDict(const char *value);

    void step();
    void learn();

    void bufferStoreTransition();
    void bufferFinishPath();

private:
    PyObject *pModule;
    PyObject *pAgent;
    PyObject *pClass;
    PythonWrapper &wrapper;

    // neural network parameters
    int obs_size = 4;
    int act_size = 3;
    double lower_bound = -1.0;
    double upper_bound = 1.0;
};

#endif /* IOV_SIM_UTIL_AGENTWRAPPER_H_ */
