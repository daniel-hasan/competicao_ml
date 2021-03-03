'''
Created on 12 de ago de 2018

@author: profhasan
'''
import os
import shutil
import subprocess
import tempfile
import uuid

import unidecode

from uncompress_data import CompressedFile, UncompatibleTypeError


class MoodleHandin(object):
    '''
    classdocs
    '''


    def __init__(self, file_pointer,dir = tempfile.gettempdir(),isUnique=False):
        '''
        Constructor
        '''
        self.path_handins = self.organiza_entrega_por_aluno(file_pointer,dir=dir)

        if(not isUnique):
            self.dictPathPorAluno = self.get_pastas_por_aluno(self.path_handins)
        else:
            self.dictPathPorAluno = {"unique":self.path_handins}

    def preproc_nome(self,nome):
        return unidecode.unidecode(nome.lower())
    def has_aluno(self,nome):
        nomePreproc = self.preproc_nome(nome)
        return nomePreproc in self.dictPathPorAluno

    def get_pasta_por_aluno(self,nome):
        nomePreproc = self.preproc_nome(nome)
        return self.dictPathPorAluno[nomePreproc]



    def uncompressAlunoDir(self,strNomAluno,strDirToSend):
        strPath = self.get_pasta_por_aluno(strNomAluno)
        self.uncompressPerPath(strPath,strDirToSend)
    def uncompressPerPath(self,strPath,strDirToSend):
        for strFile in os.listdir(strPath):
            strCompletePath = strPath+"/"+strFile
            
            if(strCompletePath != strDirToSend+"/"+strFile):
                shutil.copyfile(strCompletePath, strDirToSend+"/"+strFile)
        for strFile in os.listdir(strPath):
            strCompletePath = strPath+"/"+strFile
            #if(strFile.endswith(".7z") or strFile.endswith(".zip") or strFile.endswith(".rar") or strFile.endswith(".gz") or strFile.endswith(".tar") or strFile.endswith(".tar.gz")):
            subprocess.call(['file-roller', '--extract-to='+strDirToSend, strCompletePath])
    def del_compressed_files(self,strPath):
        for strFile in os.listdir(strPath):
            strCompletePath = strPath+"/"+strFile
            if(strFile.endswith(".7z") or strFile.endswith(".zip") or strFile.endswith(".rar") or strFile.endswith(".gz") or strFile.endswith(".tar") or strFile.endswith(".tar.gz")):
                os.remove(strCompletePath)
    def get_pastas_por_aluno(self,strPrefix):

        dicNomeAluno = {}

        for strDirName  in os.listdir( strPrefix ):
            #print(strDirName)
            #cria o diretorio dentro do tmp
            include_stu = False
            strStu_path = strPrefix+"/"+strDirName
            if(os.path.isdir(strPrefix+"/"+strDirName)):

                include_stu = True
            elif(strDirName.endswith(".7z") or strDirName.endswith(".zip") or strDirName.endswith(".rar") or strDirName.endswith(".gz") or strDirName.endswith(".tar") or strDirName.endswith(".tar.gz")):
                strDirStuName = strPrefix+"/"+strDirName.split(".")[0]
                os.makedirs(strDirStuName,exist_ok=True)
                subprocess.call(['file-roller', '--extract-to='+strDirStuName, strPrefix+"/"+strDirName])
                include_stu = True
                strStu_path = strDirStuName
            if(include_stu):
                nomAluno = self.preproc_nome(strDirName.split("_")[0])
                dicNomeAluno[nomAluno] = strStu_path

        return dicNomeAluno

    def organiza_entrega_por_aluno(self,file_pointer,dir = tempfile.gettempdir()):
        #abre o zip e salva em um dir tmp
        strDirPrefix = dir+"/"+str(uuid.uuid1())
        try:
            print("ARQUIVO AQUI: "+str(file_pointer.name)+" Salvo em: "+strDirPrefix)
            objFileZip = CompressedFile.get_compressed_file(file_pointer)
            objFileZip.descomprime_files(strDirPrefix)
        except UncompatibleTypeError:
            shutil.copy(file_pointer.name, strDirPrefix)


        return strDirPrefix


if __name__ == "__main__":
    with open("/home/profhasan/Downloads/entrega.zip","rb") as zip:

        objEntrega = MoodleHandin(zip)
        objEntrega.uncompressAlunoDir("Arthur Coelho de Souza","/tmp/dir")
