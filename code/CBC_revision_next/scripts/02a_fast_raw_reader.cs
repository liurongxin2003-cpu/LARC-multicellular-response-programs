using System;
using System.IO;
using System.IO.Compression;
using System.Diagnostics;
public static class TargetRawReader {
 public static void Run(string input,string lookupFile,string output,string totalFile,string nnzFile,int genes,int cells) {
  var watch=Stopwatch.StartNew();var lookup=Array.ConvertAll(File.ReadAllLines(lookupFile),int.Parse);
  ulong[] selected=new ulong[genes*cells],total=new ulong[cells],nnz=new ulong[cells];
  using(var file=File.OpenRead(input)) using(var gz=new GZipStream(file,CompressionMode.Decompress)) {
   string line="";int c;
   do {line="";while((c=gz.ReadByte())!=-1 && c!=10)line+=(char)c;} while(line.StartsWith("%"));
   var dim=line.Trim().Split(' ',StringSplitOptions.RemoveEmptyEntries);
   if(int.Parse(dim[0])!=lookup.Length || int.Parse(dim[1])!=cells)throw new Exception("dimension mismatch");
   long expected=long.Parse(dim[2]),seen=0;byte[] buffer=new byte[4*1024*1024];ulong value=0;int phase=0,gene=0,cell=0,n;
   while((n=gz.Read(buffer,0,buffer.Length))>0) {
    for(int i=0;i<n;i++) {byte b=buffer[i];if(b>=48 && b<=57){value=value*10+(ulong)(b-48);continue;}
     if(b!=32 && b!=10 && b!=13 && b!=9)throw new Exception("non-integer count token in raw UMI matrix");
     if(value==0 && phase==0)continue;
     if(phase==0){gene=(int)value-1;phase=1;} else if(phase==1){cell=(int)value-1;phase=2;} else {
      total[cell]+=value;nnz[cell]++;int k=lookup[gene];if(k>=0)selected[k*cells+cell]+=value;phase=0;seen++;
      if(seen%50000000==0)Console.WriteLine("Fast raw extraction "+seen+" / "+expected+"; "+watch.Elapsed.TotalSeconds.ToString("F1")+" s");
     } value=0;
    }
   }
   if(phase!=0 || seen!=expected)throw new Exception("incomplete matrix "+seen+" expected "+expected);
   Console.WriteLine("Verified integer UMI tuples: "+seen);
  }
  byte[] b1=new byte[selected.Length*8];Buffer.BlockCopy(selected,0,b1,0,b1.Length);File.WriteAllBytes(output,b1);
  byte[] b2=new byte[total.Length*8];Buffer.BlockCopy(total,0,b2,0,b2.Length);File.WriteAllBytes(totalFile,b2);Buffer.BlockCopy(nnz,0,b2,0,b2.Length);File.WriteAllBytes(nnzFile,b2);
  Console.WriteLine("Finished "+watch.Elapsed.TotalSeconds.ToString("F1")+" seconds");
 }
}
