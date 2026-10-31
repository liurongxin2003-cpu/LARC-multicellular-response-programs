using System;using System.IO;using System.Collections.Generic;
public static class SpatialTargetRows {
 public static void Extract(string input,string output){
  var targets=new HashSet<string>(new string[]{"LTA","TNFRSF1A","TNFRSF14","VWF","STAB2","BMP8A","TGFBR1","HLA-B","KIR2DL3"});
  using(var reader=new StreamReader(input))using(var writer=new StreamWriter(output)){
   writer.WriteLine(reader.ReadLine());string line;int rows=0,kept=0;
   while((line=reader.ReadLine())!=null){rows++;int tab=line.IndexOf('\t');if(tab>0 && targets.Contains(line.Substring(0,tab).ToUpperInvariant())){writer.WriteLine(line);kept++;}}
   Console.WriteLine(input+": "+rows+" gene rows; "+kept+" target rows");
  }
 }
}
