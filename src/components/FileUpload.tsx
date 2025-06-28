import React, { useState, useCallback } from 'react';
import { Upload, File, CheckCircle, AlertCircle, Loader, X } from 'lucide-react';
import type { UploadedFile } from '../types';

interface FileUploadProps {
  onFilesUploaded: (files: UploadedFile[]) => void;
  uploadedFiles: UploadedFile[];
}

const FileUpload: React.FC<FileUploadProps> = ({ onFilesUploaded, uploadedFiles }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const processFiles = async (files: FileList) => {
    setIsUploading(true);
    const newFiles: UploadedFile[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.type === 'application/pdf') {
        const newFile: UploadedFile = {
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          size: file.size,
          uploadedAt: new Date().toISOString(),
          status: 'uploading',
        };
        newFiles.push(newFile);
      }
    }

    // Update with new files immediately
    const allFiles = [...uploadedFiles, ...newFiles];
    onFilesUploaded(allFiles);

    // Simulate upload and processing
    for (const file of newFiles) {
      // Simulate upload progress
      await new Promise(resolve => setTimeout(resolve, 1000));
      file.status = 'processing';
      onFilesUploaded([...uploadedFiles, ...newFiles]);

      // Simulate processing
      await new Promise(resolve => setTimeout(resolve, 2000));
      file.status = 'completed';
      file.chunks = Math.floor(Math.random() * 50) + 10;
      onFilesUploaded([...uploadedFiles, ...newFiles]);
    }

    setIsUploading(false);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      processFiles(files);
    }
  }, [uploadedFiles, onFilesUploaded]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
  };

  const removeFile = (fileId: string) => {
    const updatedFiles = uploadedFiles.filter(file => file.id !== fileId);
    onFilesUploaded(updatedFiles);
  };

  const getStatusIcon = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <Loader className="w-5 h-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <File className="w-5 h-5 text-gray-400" />;
    }
  };

  const getStatusText = (status: UploadedFile['status']) => {
    switch (status) {
      case 'uploading':
        return 'جاري الرفع...';
      case 'processing':
        return 'جاري المعالجة...';
      case 'completed':
        return 'مكتمل';
      case 'error':
        return 'خطأ';
      default:
        return '';
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="bg-white rounded-xl shadow-lg p-8 mb-8">
      <h2 className="text-2xl font-bold text-gray-800 mb-6 text-right" dir="rtl">
        رفع ملفات PDF
      </h2>
      
      <div
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-all duration-300 ${
          isDragOver
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p className="text-xl text-gray-600 mb-4" dir="rtl">
          اسحب ملفات PDF هنا أو اضغط للتحديد
        </p>
        <p className="text-gray-500 mb-6" dir="rtl">
          يدعم النظام ملفات PDF باللغة العربية حتى 50 ميجابايت
        </p>
        <input
          type="file"
          multiple
          accept=".pdf"
          onChange={handleFileSelect}
          className="hidden"
          id="file-upload"
          disabled={isUploading}
        />
        <label
          htmlFor="file-upload"
          className={`inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg font-medium cursor-pointer transition-colors duration-200 ${
            isUploading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'
          }`}
        >
          <Upload className="w-5 h-5 ml-2" />
          اختيار الملفات
        </label>
      </div>

      {uploadedFiles.length > 0 && (
        <div className="mt-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4 text-right" dir="rtl">
            الملفات المرفوعة ({uploadedFiles.length})
          </h3>
          <div className="space-y-3">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className="flex items-center justify-between p-4 bg-gray-50 rounded-lg"
                dir="rtl"
              >
                <div className="flex items-center space-x-3 space-x-reverse">
                  {getStatusIcon(file.status)}
                  <div>
                    <p className="font-medium text-gray-800">{file.name}</p>
                    <div className="flex items-center space-x-4 space-x-reverse text-sm text-gray-500">
                      <span>{formatFileSize(file.size)}</span>
                      <span>{getStatusText(file.status)}</span>
                      {file.chunks && (
                        <span>{file.chunks} مقطع نصي</span>
                      )}
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => removeFile(file.id)}
                  className="p-2 text-gray-400 hover:text-red-500 transition-colors duration-200"
                  disabled={file.status === 'uploading' || file.status === 'processing'}
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;