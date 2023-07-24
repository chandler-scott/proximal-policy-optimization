/*
 * PythonWrapper.h
 *
 *  Created on: Jul 16, 2023
 *      Author: chandler
 */

#ifndef IOV_SIM_UTIL_PYTHONWRAPPER_H_
#define IOV_SIM_UTIL_PYTHONWRAPPER_H_

#include <Python.h>
#include <iostream>

using namespace std;

class PythonWrapper {
public:
    static PythonWrapper& getInstance();
    static void destroyInstance();

    void printPyObjectType(PyObject* pyObject);


    PythonWrapper(const PythonWrapper&) = delete;
    PythonWrapper& operator=(const PythonWrapper&) = delete;

    PyObject* callZerosBoxSpace(int box_size, double lower_bound = -1, double upper_bound = 1);
    void loadPythonModule(const char* moduleName, PyObject*& module);

    std::string serializeStateDict(PyObject* stateDictObject);
    PyObject* deserializeStateDict(std::string stateDictString);

    PyObject* ppoModule;
    PyObject* utilModule;

private:
    PythonWrapper();
    ~PythonWrapper();

    void initializePython();
    void finalizePython();

    static PythonWrapper instance;
};

#endif /* IOV_SIM_UTIL_PYTHONWRAPPER_H_ */
