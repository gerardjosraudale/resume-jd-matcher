import React from 'react';

type Props = { children: React.ReactNode };

export default function Tag({ children }: Props) {
  return (
    <span className="inline-flex items-center px-2.5 py-1 rounded-full text-xs bg-gray-100 text-gray-800 mr-2 mb-2">
      {children}
    </span>
  );
}
