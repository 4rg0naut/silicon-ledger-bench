import Foundation

CLI.args = CommandLine.arguments
let code = await CLI.runAll()
exit(code)
