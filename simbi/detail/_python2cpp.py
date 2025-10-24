import ast


class PythonToCpp(ast.NodeVisitor):
    def __init__(self):
        self.cpp_code = ""

    def visit_FunctionDef(self, node):
        args = ", ".join(f"double {arg.arg}" for arg in node.args.args)
        self.cpp_code += f"double {node.name}({args}) {{\n"
        self.generic_visit(node)
        self.cpp_code += "}\n"

    def visit_Lambda(self, node):
        args = ", ".join(f"double {arg.arg}" for arg in node.args.args)
        self.cpp_code += f"[=]({args}) -> double {{ return "
        self.visit(node.body)
        self.cpp_code += "; }"

    def visit_Return(self, node):
        self.cpp_code += "return "
        self.visit(node.value)
        self.cpp_code += ";\n"

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.cpp_code += f" {self.get_op(node.op)} "
        self.visit(node.right)

    def visit_Name(self, node):
        self.cpp_code += node.id

    def visit_Num(self, node):
        self.cpp_code += str(node.n)

    def get_op(self, op):
        if isinstance(op, ast.Add):
            return "+"
        elif isinstance(op, ast.Sub):
            return "-"
        elif isinstance(op, ast.Mult):
            return "*"
        elif isinstance(op, ast.Div):
            return "/"
        else:
            raise NotImplementedError(f"Unsupported operator: {type(op)}")
