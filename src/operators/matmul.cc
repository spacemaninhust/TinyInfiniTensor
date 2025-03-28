#include "operators/matmul.h"

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A = inputs[0]->getDims();
        auto B = inputs[1]->getDims();
        if (transA) {
            std::swap(A[A.size() - 1], A[A.size() - 2]);
        }
        if (transB) {
            std::swap(B[B.size() - 1], B[B.size() - 2]);
        }
        Shape resShape;
        for (size_t i = 0; i < A.size() - 2; i++) {
            resShape.push_back(A[i]);
        }
        resShape.push_back(A[A.size() - 2]);
        resShape.push_back(B[B.size() - 1]);
        return std::vector<Shape>{resShape};
    }

} // namespace infini