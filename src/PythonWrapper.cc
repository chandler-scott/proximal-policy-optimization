//
// Copyright (C) 2006-2011 Christoph Sommer <christoph.sommer@uibk.ac.at>
//
// Documentation for these modules is at http://veins.car2x.org/
//
// SPDX-License-Identifier: GPL-2.0-or-later
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//

#include "PythonWrapper.h"
#include <unistd.h>
#include <iostream>
#include <dlfcn.h>

PythonWrapper PythonWrapper::instance;

using namespace std;

PythonWrapper& PythonWrapper::getInstance()
{
    return instance;
}

PythonWrapper::PythonWrapper()
{
    initializePython();
    loadPythonModule("ppo", ppoModule);
    loadPythonModule("util", utilModule);
}

PythonWrapper::~PythonWrapper()
{
    Py_DECREF(ppoModule);
    Py_DECREF(utilModule);
    finalizePython();
}

void PythonWrapper::initializePython()
{
    std::cout << "Initializing Python interpreter!" << std::endl;
    Py_Initialize();

    char cwd[256]; // Buffer to hold the current working directory

    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        std::cout << "Current working directory: " << cwd << std::endl;
    }
    else {
        std::cout << "Failed to get the current working directory." << std::endl;
    }
}

void PythonWrapper::finalizePython()
{
    std::cout << "Closing Python interpreter!" << std::endl;
    Py_Finalize();
}

void PythonWrapper::destroyInstance()
{
    instance.~PythonWrapper(); // Destruct the instance
}

void PythonWrapper::printPyObjectType(PyObject* pyObject) {
    PyObject* strObject = PyObject_Str(pyObject);
    if (strObject) {
        const char* str = PyUnicode_AsUTF8(strObject);
        if (str) {
            std::cout << "Type of pyObject: " << str << std::endl;
        }
        else {
            std::cerr << "Error: Unable to convert PyObject to string." << std::endl;
        }
        Py_DECREF(strObject);
    }
    else {
        std::cerr << "Error: Unable to convert PyObject to string." << std::endl;
    }
}

void PythonWrapper::loadPythonModule(const char* moduleName, PyObject*& module)
{
    // Load the Python module using PyImport_ImportModule
    dlopen("libpython3.8.so", RTLD_LAZY | RTLD_GLOBAL);
    module = PyImport_ImportModule(moduleName);
    if (module == nullptr) {
        PyErr_Print();
        std::cerr << "Failed to load Python module - " << moduleName << std::endl;
    }
}

PyObject* PythonWrapper::callZerosBoxSpace(int box_size, double lower_bound, double upper_bound)
{
    // Convert C++ integers and doubles to Python objects
    PyObject* py_box_size = PyLong_FromLong(box_size);
    PyObject* py_lower_bound = PyFloat_FromDouble(lower_bound);
    PyObject* py_upper_bound = PyFloat_FromDouble(upper_bound);

    // Get the function object from the module
    PyObject* pFunc = PyObject_GetAttrString(utilModule, "zeros_box_space");

    if (pFunc && PyCallable_Check(pFunc)) {
        // Call the Python function with the provided arguments

        PyObject* pArgs = PyTuple_Pack(3, py_box_size, py_lower_bound, py_upper_bound);

        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);

        Py_DECREF(pArgs);

        // Decrease the reference count of the function object
        Py_XDECREF(pFunc);

        // Return the result of the Python function call
        return pResult;
    }
    else {
        if (PyErr_Occurred())
            PyErr_Print();
        fprintf(stderr, "Cannot find function 'zeros_box_space'.\n");
    }

    // If there was an error, return nullptr
    return nullptr;
}



std::string PythonWrapper::serializeStateDict(PyObject* stateDictObject)
{
    PyObject* pFunc = PyObject_GetAttrString(utilModule, "serialize_state_dict");
    if (pFunc && PyCallable_Check(pFunc)) {
            // Call the Python function with the provided arguments

            PyObject* pArgs = PyTuple_Pack(1, stateDictObject);

            PyObject* pResult = PyObject_CallObject(pFunc, pArgs);

            Py_DECREF(pArgs);

            // Decrease the reference count of the function object
            Py_XDECREF(pFunc);

            // Return the result of the Python function call
            std::string result = PyBytes_AsString(pResult);
            Py_DECREF(pResult);

            return result;
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function 'zeros_box_space'.\n");
        }

        // If there was an error, return nullptr
        return nullptr;

}



PyObject* PythonWrapper::deserializeStateDict(std::string stateDictString)
{
    std::cout << "1" << std::endl;

    PyObject* pFunc = PyObject_GetAttrString(utilModule, "bytes_to_state_dict");
    if (pFunc && PyCallable_Check(pFunc)) {
            // Call the Python function with the provided arguments

            std::cout << "2" << std::endl;
            PyObject* pyString =
                    PyBytes_FromStringAndSize(stateDictString.c_str(), stateDictString.size());
            const char* cString = PyBytes_AsString(pyString);

            std::cout << cString << std::endl;

            if (pyString != NULL) {
                PyObject* pArgs = PyTuple_Pack(1, pyString);
                std::cout << "3" << std::endl;
                PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                std::cout << "4" << std::endl;
                Py_DECREF(pArgs);

                // Decrease the reference count of the function object
                Py_XDECREF(pFunc);

                // Return the result of the Python function call
                return pResult;
            }
            else
            {
                PyErr_Print();
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function 'zeros_box_space'.\n");
        }

        // If there was an error, return nullptr
        return nullptr;

}
