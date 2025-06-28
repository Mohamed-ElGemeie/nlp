import type { RAGResponse, SystemStats, UploadedFile, SearchResult } from '../types';

// Mock system statistics
export const mockSystemStats: SystemStats = {
  totalDocuments: 156,
  totalChunks: 2847,
  indexSize: 15728640, // ~15MB
  lastUpdated: new Date().toISOString(),
  processingStatus: 'ready'
};

// Mock search results
export const mockSearchResults: SearchResult[] = [
  {
    content: `المادة (45): تخضع الشركات المساهمة للضريبة على الأرباح التجارية والصناعية وفقاً للأحكام المنصوص عليها في قانون الضرائب على الدخل. وتحدد نسبة الضريبة بـ 22.5% من صافي الأرباح السنوية للشركة بعد خصم جميع المصروفات والاستهلاكات المقررة قانوناً.

كما تلتزم الشركة بتقديم الإقرار الضريبي السنوي خلال أربعة أشهر من انتهاء السنة المالية، مرفقاً بالميزانية العمومية وحساب الأرباح والخسائر مصدقة من مراقب حسابات مقيد بسجل مراقبي الحسابات.`,
    source: 'قانون الشركات المصري.pdf',
    metadata: 'قانون الشركات المصري.pdf - Chunk 23',
    similarity_score: 0.924
  },
  {
    content: `المادة (78): تستحق الضريبة على الدخل من الأشخاص الاعتبارية بمعدل 22.5% من صافي الدخل الخاضع للضريبة. ويعتبر صافي الدخل الخاضع للضريبة هو الدخل المحقق من مزاولة النشاط التجاري أو الصناعي أو المهني بعد خصم جميع التكاليف واجبة الخصم وفقاً لأحكام هذا القانون.

وتطبق ضريبة إضافية قدرها 5% على الدخل الذي يزيد عن 10 مليون جنيه سنوياً، بحيث يصبح المعدل الإجمالي للضريبة 27.5% على الشريحة الزائدة.`,
    source: 'قانون الضرائب على الدخل.pdf',
    metadata: 'قانون الضرائب على الدخل.pdf - Chunk 45',
    similarity_score: 0.891
  },
  {
    content: `المادة (12): على كل شركة مساهمة أن تؤدي ضريبة الدمغة على رأس المال المدفوع بمعدل 0.15% من قيمة رأس المال، وذلك عند التأسيس وعند كل زيادة في رأس المال.

كما تخضع الشركة لضريبة الدمغة على العقود والمستندات وفقاً للجدول المرفق بقانون الدمغة، وتلتزم بأداء الضريبة العقارية على العقارات المملوكة لها وفقاً لأحكام قانون الضريبة العقارية.`,
    source: 'قانون الدمغة والتسجيل.pdf',
    metadata: 'قانون الدمغة والتسجيل.pdf - Chunk 8',
    similarity_score: 0.823
  }
];

// Mock AI response
export const mockAIResponse = `بناءً على النصوص القانونية المتوفرة، يمكنني توضيح الضرائب المفروضة على الشركات في القانون المصري كالتالي:

## الضرائب الأساسية على الشركات:

### 1. ضريبة الأرباح التجارية والصناعية:
- **المعدل الأساسي**: 22.5% من صافي الأرباح السنوية
- **المعدل الإضافي**: 5% على الدخل الذي يزيد عن 10 مليون جنيه سنوياً
- **المعدل الإجمالي للشريحة العليا**: 27.5%

**المرجع القانوني**: المادة (45) من قانون الشركات المصري، والمادة (78) من قانون الضرائب على الدخل.

### 2. ضريبة الدمغة على رأس المال:
- **المعدل**: 0.15% من قيمة رأس المال المدفوع
- **التوقيت**: عند التأسيس وعند كل زيادة في رأس المال

**المرجع القانوني**: المادة (12) من قانون الدمغة والتسجيل.

### 3. ضرائب إضافية:
- **ضريبة الدمغة على العقود والمستندات**: وفقاً للجدول المرفق بقانون الدمغة
- **الضريبة العقارية**: على العقارات المملوكة للشركة

## الالتزامات الضريبية:

1. **تقديم الإقرار الضريبي السنوي**: خلال 4 أشهر من انتهاء السنة المالية
2. **المستندات المطلوبة**: الميزانية العمومية وحساب الأرباح والخسائر مصدقة من مراقب حسابات مقيد

هذه هي الضرائب الأساسية المنصوص عليها في النصوص القانونية المتوفرة، وقد تكون هناك ضرائب أخرى غير مذكورة في المصادر المتاحة حالياً.`;

// Function to simulate search with realistic delay
export const simulateSearch = async (query: string): Promise<RAGResponse> => {
  // Simulate processing delay
  await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000));
  
  // Simulate some failures occasionally
  if (Math.random() < 0.05) {
    return {
      question: query,
      retrieved_documents: [],
      model_answer: '',
      error: 'حدث خطأ في الاتصال بالنظام. يرجى المحاولة مرة أخرى.'
    };
  }

  // Filter results based on query relevance (simple keyword matching)
  const queryLower = query.toLowerCase();
  const relevantResults = mockSearchResults.filter(result => 
    result.content.includes('ضريبة') || 
    result.content.includes('شركة') ||
    queryLower.includes('ضريبة') ||
    queryLower.includes('شركة')
  );

  return {
    question: query,
    retrieved_documents: relevantResults.length > 0 ? relevantResults : mockSearchResults,
    model_answer: mockAIResponse
  };
};

// Mock uploaded files
export const mockUploadedFiles: UploadedFile[] = [
  {
    id: '1',
    name: 'قانون الشركات المصري.pdf',
    size: 2548372,
    uploadedAt: '2024-01-15T10:30:00Z',
    status: 'completed',
    chunks: 45
  },
  {
    id: '2',
    name: 'قانون الضرائب على الدخل.pdf',
    size: 1923847,
    uploadedAt: '2024-01-15T11:15:00Z',
    status: 'completed',
    chunks: 38
  },
  {
    id: '3',
    name: 'قانون العمل الموحد.pdf',
    size: 3471925,
    uploadedAt: '2024-01-15T12:00:00Z',
    status: 'completed',
    chunks: 67
  }
];