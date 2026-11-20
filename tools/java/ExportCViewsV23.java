package com.gsat.tools;
import java.nio.file.Files;
import java.nio.file.Path;
import org.json.JSONArray;
import org.json.JSONObject;
import ghidra.GhidraJarApplicationLayout;
import ghidra.framework.Application;
import ghidra.framework.HeadlessGhidraApplicationConfiguration;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileOptions;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.Program;
import ghidra.app.util.opinion.ElfLoader;
import ghidra.util.task.TaskMonitor;

public class ExportCViewsV23 {
    public static void main(String[] args) throws Exception {
        Application.initializeApplication(new GhidraJarApplicationLayout(),
                new HeadlessGhidraApplicationConfiguration());
        BaseTool tool=ToolFactory.initTool(new String[]{"pcode-extractor-v2", "-m","elf",
            "-f",args[0],"-c",args[1],"-of","SOG","-opt","0","-p",args[3],"-n","cexport"});
        if(tool==null) throw new IllegalStateException("ELF loader failed");
        DecompInterface decompiler=new DecompInterface();
        JSONObject output=new JSONObject();
        try {
            DecompileOptions options=new DecompileOptions();
            decompiler.setOptions(options);
            decompiler.toggleCCode(true);
            if(!decompiler.openProgram(tool.program)) throw new IllegalStateException("Decompiler open failed");
            JSONObject wrapper=new JSONObject(Files.readString(Path.of(args[1])));
            JSONArray rows=wrapper.getJSONArray(wrapper.keys().next());
            String originalBase=tool.program.getOptions(Program.PROGRAM_INFO)
                .getString(ElfLoader.ELF_ORIGINAL_IMAGE_BASE_PROPERTY,"0x0");
            long offset=tool.program.getImageBase().getOffset()-Long.decode(originalBase);
            for(int i=0;i<rows.length();i++) {
                JSONObject row=rows.getJSONObject(i);
                String key=row.getString("start_ea");
                Address address=tool.program.getAddressFactory().getAddress(key).add(offset);
                Function function=tool.program.getFunctionManager().getFunctionAt(address);
                JSONObject item=new JSONObject();
                item.put("load_offset",offset).put("loaded_address",address.toString());
                if(function==null) {
                    item.put("status","unavailable").put("reason","no_function_at_exact_address");
                } else {
                    DecompileResults result=decompiler.decompileFunction(function,60,TaskMonitor.DUMMY);
                    if(result.decompileCompleted() && result.getDecompiledFunction()!=null) {
                        item.put("status","ok").put("code",result.getDecompiledFunction().getC());
                    } else {
                        item.put("status","unavailable").put("reason",result.getErrorMessage());
                    }
                }
                output.put(key,item);
                Files.writeString(Path.of(args[2]),output.toString(2)+"\n");
            }
        } finally {
            decompiler.dispose();
            tool.close();
        }
    }
}
