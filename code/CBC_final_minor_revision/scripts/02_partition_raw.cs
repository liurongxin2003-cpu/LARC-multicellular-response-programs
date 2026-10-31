using System;using System.IO;using System.IO.Compression;using System.Diagnostics;
public static class LibraryPartitionReader {
 public static void Run(string input,string lookup,string output,int groups,int cells){
  var sw=Stopwatch.StartNew();var lines=File.ReadAllLines(lookup);if(lines.Length!=cells)throw new Exception("cell count mismatch");
  int[] group=new int[cells],local=new int[cells];for(int i=0;i<cells;i++){var a=lines[i].Split('\t');group[i]=int.Parse(a[0]);local[i]=int.Parse(a[1]);}
  BinaryWriter[] writers=new BinaryWriter[groups];long[] nnz=new long[groups],total=new long[cells],cellNnz=new long[cells];
  try{
   for(int i=0;i<groups;i++)writers[i]=new BinaryWriter(new FileStream(Path.Combine(output,"library_"+i.ToString("D3")+".bin"),FileMode.Create,FileAccess.Write,FileShare.Read,262144));
   using(var fs=File.OpenRead(input))using(var gz=new GZipStream(fs,CompressionMode.Decompress)){
    string line="";int c;do{line="";while((c=gz.ReadByte())!=-1&&c!=10)line+=(char)c;}while(line.StartsWith("%"));
    var dim=line.Trim().Split(' ',StringSplitOptions.RemoveEmptyEntries);if(int.Parse(dim[1])!=cells)throw new Exception("source dimension mismatch");long expected=long.Parse(dim[2]),seen=0,value=0;int phase=0,gene=0,cell=0,n;bool token=false;byte[] buffer=new byte[4*1024*1024];
    while((n=gz.Read(buffer,0,buffer.Length))>0)for(int k=0;k<n;k++){
     byte b=buffer[k];if(b>=48&&b<=57){value=value*10+b-48;token=true;continue;}if(b!=32&&b!=10&&b!=13&&b!=9)throw new Exception("noninteger source token");if(!token)continue;
     if(phase==0){gene=checked((int)value)-1;phase=1;}else if(phase==1){cell=checked((int)value)-1;phase=2;}else{int g=group[cell];if(g>=0){writers[g].Write(local[cell]);writers[g].Write(gene);writers[g].Write(checked((int)value));nnz[g]++;total[cell]+=value;cellNnz[cell]++;}seen++;phase=0;}value=0;token=false;
    }
    if(phase!=0||token||seen!=expected)throw new Exception("incomplete source matrix");Console.WriteLine("Scanned "+seen+" integer records in "+sw.Elapsed.TotalSeconds.ToString("F1")+" seconds");
   }
  }finally{foreach(var w in writers)if(w!=null)w.Dispose();}
  using(var w=new StreamWriter(Path.Combine(output,"selected_cell_count_validation.tsv"))){w.WriteLine("source_cell_index\ttotal_counts\tnnz");for(int i=0;i<cells;i++)if(group[i]>=0)w.WriteLine(i+"\t"+total[i]+"\t"+cellNnz[i]);}
  for(int i=0;i<groups;i++)Console.WriteLine("Library "+i+" nnz="+nnz[i]);
 }
}
